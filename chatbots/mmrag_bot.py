import uuid
from fastapi import HTTPException
from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from chatbots.qa_bot import LLMService, EmbeddingService

from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.memory import ConversationSummaryMemory
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

class MMRagBotState(TypedDict):
    question: str
    history: Annotated[list[AnyMessage], add_messages]
    last_response_refs: dict

class RAG_Bot:
    def __init__(self, text_llm=LLMService('AzureOpenAI', 'DeepSeek-R1-houab', max_gen_tokens=1024), multimodal_llm=LLMService('Google', 'gemini-2.0-flash'), embedding=EmbeddingService()):
        self.text_llm = text_llm
        self.multimodal_llm = multimodal_llm
        self.embedding = embedding
        self.file_path = None
        self.title = None
        self.db_tokens = []
        self.vector_db = None
        self.retriever = None
        self.memory = ConversationSummaryMemory(llm=self.text_llm.model, memory_key="chat_history", return_messages=True)

        graph = StateGraph(MMRagBotState)
        graph.add_node("choose_llm", self.choose_llm)
        graph.add_node("retrieve_doc", self.retrieve_doc)

        graph.add_edge(START, "choose_llm")
        graph.add_conditional_edges("choose_llm", self.should_retrieve)
        graph.add_edge('retrieve_doc', END)

        # Compile the graph into a runnable
        self.graph = graph.compile(checkpointer=MemorySaver())

    def choose_llm(self, state):
        system_prompt = f"""
        You are an AI assistant that determines whether there is a need to retrieve (using RAG) a document (if there is a title) to answer the user's prompt.
        - If there is a need to retrieve and there is a title, you call the "retrieve_doc" with a message that is easy for the retriever retrieving the vector database tool and doesn't say anything.
        - If there is a need to retrieve and no title, you tell the user to upload the document by calling "SubmitFinalAnswer".
        - If there is no need to retrieve, you answer the question directly by calling "SubmitFinalAnswer".

        Your output can be the "retrieve_doc" or "SubmitFinalAnswer" tool.
        Remember to answer only when you know the answer is true, and don't make fake information. If you don't know the answer, say, "I don't know".
        Title: {self.title}
        """
        self.llm_choosing_chain = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{messages}")]) | LLMService('AzureOpenAI', 'gpt-35-turbo-16k').model.bind_tools([self.retrieve_doc, self.SubmitFinalAnswer])
        response = self.llm_choosing_chain.invoke({"messages": [state['question']]})
        if not getattr(response, "tool_calls", None):
            return {'history': [response], 'last_response_refs': {'context': 'There is no use RAG.'}}

        return {'history': [response],}
    
    def should_retrieve(self, state):
        if getattr(state['history'][-1], "tool_calls", None):
            if state['history'][-1].tool_calls[0]["name"] == "retrieve_doc":
                return "retrieve_doc"

        return END

    class SubmitFinalAnswer(BaseModel):
        """Submit the final answer to the user based on the query results."""
        final_answer: str = Field(..., description="The final answer to the user")

    def clear_memory(self):
        self.memory.clear()

    def extract_doc(self, file_path, chunking_strategy="by_title", max_characters=10000, combine_text_under_n_chars=2000, new_after_n_chars=6000):
        try:
            # Document partitioning: https://docs.unstructured.io/open-source/core-functionality/chunking
            chunks = partition_pdf(
                filename=self.file_path,
                infer_table_structure=True,            # extract tables
                strategy="hi_res",                     # mandatory to infer tables
                extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
                # image_output_dir_path=output_path,   # if None, images and tables will saved in base64
                extract_image_block_to_payload=True,   # if true, will extract base64 for API usage
                chunking_strategy=chunking_strategy,   # or 'basic'
                max_characters=max_characters,         # defaults to 500
                combine_text_under_n_chars=combine_text_under_n_chars,  # defaults to 0
                new_after_n_chars=new_after_n_chars,
                # extract_images_in_pdf=True,          # deprecated
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Separate tables from texts
        texts = []
        for chunk in chunks:
            if "CompositeElement" in str(type((chunk))):
                chunk_els = chunk.metadata.orig_elements
                texts.append({'page_number': chunk.metadata.page_number, 'text': ''})

                for i in range(len(chunk_els)):
                    if "Table" not in str(type(chunk_els[i])) and "Image" not in str(type(chunk_els[i])):
                        texts[-1]['text'] += chunk_els[i].text + "\n"
                
        # Get the images from the CompositeElement objects
        def get_img_tab(chunks):
            images = []
            tables = []
            for chunk in chunks:
                if "CompositeElement" in str(type(chunk)):
                    chunk_els = chunk.metadata.orig_elements

                    i = 0
                    while i < len(chunk_els):
                        if "Image" in str(type(chunk_els[i])):
                            images.append({'page_number': [chunk_els[i].metadata.page_number], 'caption': '', 'base64': [chunk_els[i].metadata.image_base64]})

                            j = i + 1
                            while j < len(chunk_els) and "Image" in str(type(chunk_els[j])):
                                images[-1]['page_number'].append(chunk_els[j].metadata.page_number)
                                images[-1]['base64'].append(chunk_els[j].metadata.image_base64)
                                j += 1

                            i = j
                            if i < len(chunk_els) and ('NarrativeText' in str(type(chunk_els[i])) or 'FigureCaption' in str(type(chunk_els[i]))):
                                images[-1]['caption'] = chunk_els[i].text

                        if "Table" in str(type(chunk_els[i])):
                            tables.append({'page_number': chunk_els[i].metadata.page_number, 'caption': '', 'html': chunk_els[i].metadata.text_as_html})
                            if i - 1 >= 0 and ('NarrativeText' in str(type(chunk_els[i-1])) or 'FigureCaption' in str(type(chunk_els[i-1]))):
                                tables[-1]['caption'] = chunk_els[i-1].text

                        i += 1

            return images, tables

        images, tables = get_img_tab(chunks)
        return texts, images, tables

    def get_doc_title(self, first_text):
        prompt_text = [("system", """
        Based on the first part of a paper. Guess the title of the paper."
        Respond only with the title, no additionnal comment.
        Do not start your message by saying "Here is a title" or anything like that.
        Just give the title as it is.
        """), ("human", "{input}")]

        prompt = ChatPromptTemplate(prompt_text)
        chain = prompt | self.text_llm.model
        self.title = chain.invoke(first_text['text']).content.split("</think>\n\n", 1)[-1]

        # self.title = "Attention Is All You Need"
        # self.title = "MAT: Mask-Aware Transformer for Large Hole Image Inpainting"
    
    def summarize_texts(self, texts):
        prompt_text = [("system", f"""
        Give a concise summary of the texts. For context, the text is part of a research paper named {self.title}.
        Respond only with the summary, including which sections and subs the text includes and the content of the text.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        """), ("human", "{input}")]

        prompt = ChatPromptTemplate(prompt_text)
        chain = prompt | self.text_llm.model | StrOutputParser()

        text_summaries = chain.batch([text['text'] for text in texts], {"max_concurrency": 3})
        text_summaries = [summary.split("</think>\n\n", 1)[-1] for summary in text_summaries]

        # text_summaries = ['Title and Authors: The paper "Attention Is All You Need" is authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin, with equal contributions. Key contributions include Jakob Uszkoreit proposing self-attention to replace RNNs, Ashish Vaswani and Illia Polosukhin designing the first Transformer models, Noam Shazeer introducing scaled dot-product attention and multi-head attention, and other authors contributing to codebases, experiments, and infrastructure.  \n\nAbstract: The paper introduces the Transformer, a neural network architecture relying solely on attention mechanisms without recurrence or convolutions. It outperforms existing models on English-to-German and English-to-French machine translation tasks, achieving 28.4 and 41.8 BLEU scores, respectively, with greater parallelism and reduced training time (3.5 days on 8 GPUs). The model also generalizes effectively to English constituency parsing.  \n\nAttribution Note: Google grants permission to reproduce tables and figures with proper attribution for journalistic or scholarly works.  \n\nTechnical Header: Includes arXiv identifier (arXiv:1706.03762v7), publication date (2023), and conference details (31st NIPS 2017, Long Beach, CA, USA).',
        # 'The text includes sections 1 (Introduction) and 2 (Background) of the paper.  \n\n**1 Introduction**  \nRecurrent neural networks (RNNs), LSTMs, and GRUs are dominant in sequence modeling tasks like machine translation but suffer from sequential computation, limiting parallelization and efficiency. Attention mechanisms, often paired with RNNs, enable modeling of long-range dependencies. The authors propose the **Transformer**, a model architecture eliminating recurrence and relying solely on attention for global dependencies. This approach enables greater parallelization, achieving state-of-the-art translation quality with 12 hours of training on 8 GPUs.  \n\n**2 Background**  \nPrior work (Extended Neural GPU, ByteNet, ConvS2S) used convolutional networks to parallelize computation but faced challenges in learning distant dependencies due to growing computational costs with positional distance. The Transformer reduces this to constant operations via **self-attention** (attention mechanisms within a single sequence), though averaging attention-weighted positions may reduce resolution—addressed later by multi-head attention. Self-attention has been applied in tasks like summarization and sentence representation. The Transformer is the first transduction model to rely **entirely** on self-attention without RNNs or convolution, contrasting with earlier hybrid models. The following sections detail the architecture and advantages over existing approaches.',
        # '**Section 3: Model Architecture**  \nThe Transformer adopts an encoder-decoder structure common in neural sequence transduction models. The encoder converts an input sequence into continuous representations, while the decoder generates an output sequence auto-regressively (using prior outputs as inputs). The architecture relies on stacked self-attention and point-wise fully connected layers for both encoder and decoder (Figure 1).  \n\n**Section 3.1: Encoder and Decoder Stacks**  \n**Encoder**: Comprises 6 identical layers. Each layer has two sub-layers:  \n1. **Multi-head self-attention mechanism**.  \n2. **Position-wise feed-forward network**.  \nResidual connections and layer normalization are applied to each sub-layer: `LayerNorm(x + Sublayer(x))`. All sub-layers and embeddings output vectors of dimension *d_model = 512*.  \n\n**Decoder**: Also has 6 identical layers, with three sub-layers per layer:  \n1. **Masked multi-head self-attention** (prevents attention to subsequent positions).  \n2. **Multi-head attention over encoder outputs**.  \n3. **Position-wise feed-forward network**.  \nResidual connections and layer normalization are used similarly. The masking ensures predictions for position *i* depend only on earlier positions. Input embeddings are offset by one position to preserve auto-regressive properties.',
        # 'Section 3.2 (Attention) introduces the attention function as a mechanism mapping a query and key-value pairs to an output, computed as a weighted sum of values. The weights derive from a compatibility function between the query and keys.  \n\nSubsection 3.2.1 (Scaled Dot-Product Attention) details the authors’ specific attention variant. Inputs include queries and keys of dimension \\(d_k\\) and values of dimension \\(d_v\\). The scaled dot-product attention computes dot products of queries with all keys, scales them by \\(1/\\sqrt{d_k}\\), applies softmax for weights, and multiplies by values. For batched processing, queries, keys, and values are packed into matrices \\(Q\\), \\(K\\), and \\(V\\), with the output computed as:  \n\\[\n\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V\n\\]  \nThe text contrasts this with additive attention, noting dot-product attention’s superior speed and space efficiency due to optimized matrix operations. However, unscaled dot-product attention underperforms additive attention for large \\(d_k\\), as large dot product magnitudes push softmax into low-gradient regions. Scaling by \\(1/\\sqrt{d_k}\\) mitigates this issue. The section also references Figure 2, illustrating the scaled dot-product mechanism and multi-head attention architecture.',
        # '**3.2.2 Multi-Head Attention**: This subsection introduces multi-head attention, where queries, keys, and values are linearly projected *h* times (using learned projections) into lower-dimensional spaces (*dₖ*, *dᵥ*). Each projection undergoes parallel scaled dot-product attention, producing *dᵥ*-dimensional outputs. These are concatenated and reprojected to form the final output. Multi-head attention enables the model to focus on diverse representation subspaces simultaneously. The paper uses *h=8* heads, with *dₖ=dᵥ=64* (since *dₖ=dᵥ=d_model/h*), maintaining computational efficiency comparable to single-head attention. The mathematical formulation includes *headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)* and *MultiHead(Q,K,V) = Concat(head₁,…,headₕ)W^O*, with projection matrices *Wᵢ^Q, Wᵢ^K, Wᵢ^V, W^O*.  \n\n**3.2.3 Applications of Attention in our Model**: Describes three uses of multi-head attention in the Transformer:  \n1. **Encoder-decoder attention**: Decoder queries attend to encoder outputs (keys/values), enabling cross-sequence attention.  \n2. **Encoder self-attention**: All keys, values, and queries in the encoder come from the same source (previous encoder layer), allowing full sequence attention.  \n3. **Decoder self-attention**: Self-attention in the decoder is restricted (via masking) to prevent positions from attending to future positions, preserving autoregressive properties. Masking is applied by setting illegal softmax inputs to −∞.',
        # '**3.3 Position-wise Feed-Forward Networks**  \nDescribes the feed-forward network (FFN) in each encoder/decoder layer, applied identically to every position. The FFN comprises two linear transformations with ReLU activation:  \n- Formula: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂.  \n- Input/output dimension: *d_model* = 512; inner layer dimension: *d_ff* = 2048.  \n- Parameters differ across layers but are shared across positions, akin to 1D convolutions.  \n\n**3.4 Embeddings and Softmax**  \nDetails embedding layers and output probability conversion:  \n- Learned embeddings convert input/output tokens to *d_model*-dimensional vectors (512).  \n- Weight sharing: Embedding layers and pre-softmax linear layer share weights, scaled by √*d_model*.  \n- Softmax generates next-token probabilities from decoder output.  \n\n**3.5 Positional Encoding**  \nExplains positional encoding to inject sequence order information:  \n- Sinusoidal functions encode positions:  \n  - PE(pos, 2i) = sin(pos/10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos/10000^(2i/d_model)).  \n- Wavelengths form a geometric progression (2π to 10000·2π), enabling relative position learning.  \n- Sinusoidal encodings chosen over learned embeddings for better extrapolation to longer sequences.  \n\n**Table 1**  \nCompares layer types (e.g., self-attention, convolution) by:  \n- Maximum path lengths.  \n- Per-layer complexity.  \n- Minimum sequential operations.  \n- Variables: sequence length (*n*), dimension (*d*), kernel size (*k*), neighborhood size (*r*).',
        # 'The text is from Section 4 ("Why Self-Attention") of the research paper. It compares self-attention layers with recurrent and convolutional layers for sequence transduction tasks, focusing on three criteria: computational complexity per layer, parallelizability (measured by sequential operations required), and path length for long-range dependencies. \n\nSelf-attention layers connect all positions in a sequence with constant sequential operations (O(1)), outperforming recurrent layers (O(n) sequential steps). Computationally, self-attention is faster than recurrent layers when sequence length *n* is smaller than representation dimensionality *d*, typical in machine translation. For very long sequences, restricting self-attention to local neighborhoods (size *r*) is proposed as a future direction, increasing maximum path length to O(*n/r*). \n\nConvolutional layers require O(*n/k*) or O(logₖ(*n*)) layers to connect all positions (depending on kernel type), leading to longer dependency paths. While standard convolutions are costly (factor *k*), separable convolutions reduce complexity but still match the combined cost of self-attention and feed-forward layers. \n\nAdditionally, self-attention offers interpretability: attention heads learn diverse tasks, often aligning with syntactic and semantic structures (examples provided in the appendix). The analysis references Table 1 for complexity comparisons and emphasizes self-attention’s advantages in handling long-range dependencies via shorter network paths.',
        # '**5 Training** outlines the training methodology for the Transformer models.  \n- **5.1 Training Data and Batching**: Utilized WMT 2014 English-German (4.5M sentence pairs, 37K shared byte-pair tokens) and English-French (36M sentences, 32K word-piece tokens) datasets. Batches were grouped by sequence length, with each batch containing ~25K source and target tokens.  \n- **5.2 Hardware and Schedule**: Trained on 8 NVIDIA P100 GPUs. Base models (100K steps, 12 hours) took 0.4s/step; larger models (300K steps, 3.5 days) took 1.0s/step.  \n- **5.3 Optimizer**: Adam optimizer (β1=0.9, β2=0.98, ε=10⁻⁹) with dynamic learning rate: linearly increased for first 4k steps, then decayed proportionally to inverse square root of step number.  \n- **5.4 Regularization**:  \n  - **Residual Dropout** (rate=0.1) applied to sub-layer outputs and embedding/positional encoding sums.  \n  - **Label Smoothing** (ε=0.1) improved accuracy and BLEU despite increased perplexity.',
        # '**6 Results**  \n**6.1 Machine Translation**: Reports performance of the Transformer model on WMT 2014 English-to-German and English-to-French translation tasks. The "big" Transformer model achieves state-of-the-art BLEU scores (28.4 for English-German, 41.0 for English-French) with significantly lower training costs than previous models. Details include training time (3.5 days on 8 P100 GPUs), checkpoint averaging (last 5 checkpoints for base models, last 20 for big models), and inference hyperparameters (beam size 4, length penalty α=0.6). Comparisons to prior work in Table 2 highlight translation quality and computational efficiency.  \n\n**6.2 Model Variations**: Analyzes the impact of architectural changes to the base Transformer model on English-to-German translation (newstest2013). Experiments vary components like attention heads, attention key/value dimensions, FFN layer size, and attention mechanisms (Table 3). Metrics include per-wordpiece perplexity and BLEU scores. Configurations not explicitly listed retain base model parameters.',
        # 'The text includes sections on model experiments in machine translation (6.2) and English constituency parsing (6.3).  \n\n**6.2 Model Variations (Table 3)**:  \n- **(A)**: Varying attention heads while keeping computation constant shows single-head attention underperforms by 0.9 BLEU; too many heads degrade quality.  \n- **(B)**: Reducing attention key size \\(d_k\\) harms performance, suggesting dot-product compatibility may be insufficient.  \n- **(C-D)**: Larger models improve results, and dropout effectively prevents overfitting.  \n- **(E)**: Learned positional embeddings perform similarly to sinusoidal encoding.  \n\n**6.3 English Constituency Parsing**:  \n- A 4-layer Transformer (\\(d_{\\text{model}}=1024\\)) was trained on WSJ Penn Treebank (40K sentences) and semi-supervised data (17M sentences), with vocabularies of 16K/32K tokens. Minimal hyperparameter tuning was done (dropout, learning rates, beam size).  \n- During inference, output length was increased (input + 300), beam size 21, and \\(\\alpha=0.3\\).  \n- Results (Table 4) show the Transformer outperforms most models (except Recurrent Neural Network Grammar) and surpasses BerkeleyParser in the WSJ-only setting, demonstrating strong generalization without task-specific adjustments.',
        # '**7 Conclusion**  \n- Introduces the Transformer, a sequence transduction model relying solely on attention mechanisms, replacing recurrent layers in encoder-decoder architectures with multi-headed self-attention.  \n- Highlights faster training compared to recurrent/convolutional models and state-of-the-art results on WMT 2014 English-to-German and English-to-French translation tasks.  \n- Outlines future directions: extending the Transformer to non-text modalities (images, audio, video), investigating local/restricted attention for large inputs/outputs, and reducing sequentiality in generation.  \n- Mentions code availability on GitHub.  \n\n**Acknowledgements**  \n- Credits Nal Kalchbrenner and Stephan Gouws for contributions.  \n\n**References**  \n- Lists 35 citations (e.g., layer normalization, attention mechanisms, LSTM, residual networks, Adam optimization) supporting the research.',
        # 'The text includes references [36] to [40], citing prior works on computer vision architectures (Inception), neural machine translation systems, and parsing methods. Sections 12–15 present visualizations of the self-attention mechanisms in the Transformer model. Section 12 introduces the figures, while sections 13–15 analyze specific examples: Figure 3 (section 13) demonstrates attention heads capturing long-distance dependencies (e.g., linking "making" to "difficult" in a sentence). Figure 4 (section 14) highlights heads resolving anaphora (e.g., tracking the referent of "its"). Figure 5 (section 15) shows heads performing structurally oriented tasks (e.g., attending to syntactic or semantic sentence patterns). All examples focus on encoder self-attention in layer 5 of a 6-layer model, illustrating diverse, specialized roles of different attention heads.']

        # text_summaries = ['The text includes the **Abstract** and **Introduction** sections of the research paper "MAT: Mask-Aware Transformer for Large Hole Image Inpainting."  \n\n**Abstract**:  \nThe paper introduces **MAT**, a transformer-based model for high-resolution image inpainting that combines transformers and convolutions to efficiently handle large missing regions. It proposes a **mask-aware transformer block** where attention is dynamically restricted to valid tokens (non-masked regions) to improve fidelity and diversity. Experiments show state-of-the-art results on benchmark datasets, with code publicly released.  \n\n**Introduction (Section 1)**:  \nImage inpainting aims to fill missing regions with plausible content and has applications in editing, retargeting, and restoration. Existing methods use attention or transformers but are limited to low resolutions due to computational costs. **MAT** addresses this by unifying transformers and convolutions for high-resolution processing. The key innovation is the **inpainting-oriented transformer block** with a dynamic mask that guides attention to aggregate non-local information only from valid regions. This design enhances efficiency and output quality. The introduction highlights the model’s ability to achieve photo-realistic and diverse results on high-resolution images (e.g., real-world and dataset examples like Places and FFHQ) and emphasizes its superiority over existing methods through extensive experiments.']

        return text_summaries

    def summarize_tables(self, tables):
        prompt_text = [("system", f"""
        Give a concise summary of the tables. For context, the tables is part of a research paper named {self.title}.
        Respond only with the summary, no additionnal comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        """), ("human", "{input}")]

        prompt = ChatPromptTemplate(prompt_text)
        chain = prompt | self.text_llm.model | StrOutputParser()

        tables_input = [f"Caption: {table['caption']}\nTable:\n{table['html']}" for table in tables]
        table_summaries = chain.batch(tables_input , {"max_concurrency": 3})
        table_summaries = [summary.split("</think>\n\n", 1)[-1] for summary in table_summaries]

        # table_summaries = ['Table 1 compares computational properties of different neural network layers (self-attention, recurrent, convolutional, restricted self-attention) in terms of time complexity per layer, sequential operations required, and maximum path length between input-output positions. It quantifies efficiency tradeoffs: self-attention achieves constant sequential operations and shortest path lengths but quadratic complexity in sequence length (n), while recurrent layers have linear complexity but require O(n) sequential steps. Convolutional layers offer intermediate path lengths (logarithmic) with kernel size k, and restricted self-attention reduces complexity by limiting attention to neighborhoods of size r. Variables include d (dimension size), k (kernel size), and r (neighborhood size).',
        # "This table compares the performance and computational efficiency of various neural machine translation models, including the Transformer, on English-to-German (EN-DE) and English-to-French (EN-FR) translation tasks. It lists BLEU scores (translation quality) and training costs (in FLOPs) for each model. The Transformer (big) achieves the highest BLEU scores (28.4 EN-DE, 41.8 EN-FR) with significantly lower training costs (2.3×10¹⁹ FLOPs for EN-DE, 1×10¹⁹ FLOPs for EN-FR) compared to previous models like ConvS2S, GNMT+RL, and ByteNet. The table demonstrates the Transformer's superior efficiency and performance over recurrent and convolutional architectures.",
        # 'Table 3 compares architectural variations of the Transformer model against the base configuration for English-to-German translation (newstest2013). It evaluates changes in hyperparameters (e.g., attention heads, model dimensions, dropout rates, feed-forward layer sizes) and their impact on perplexity (per-wordpiece) and BLEU scores. Sections (A)-(E) test specific modifications: (A) varying attention head counts and key/value dimensions, (B) adjusting query/key dimensions, (C) altering layer depth, embedding size, and feed-forward width, (D) modifying dropout rates, and (E) replacing sinusoidal positional encoding with learned embeddings. A "big" model with increased capacity (6 layers, 1024-dim, 16 heads) achieves the best performance (4.33 perplexity, 26.4 BLEU). Metrics highlight trade-offs between model size, training efficiency, and translation quality.',
        # "Table 4 compares the Transformer's performance on English constituency parsing (WSJ Section 23) against prior models, grouped by training method. It lists F1 scores for parsers using discriminative (WSJ-only), semi-supervised, multi-task, and generative training. The 4-layer Transformer achieves 91.3 F1 (WSJ-only) and 92.7 F1 (semi-supervised), outperforming most discriminative/semi-supervised baselines and approaching state-of-the-art generative models (93.3 F1). The table demonstrates the Transformer's parsing generalization capability despite its simplicity."]

        # table_summaries = []

        return table_summaries

    def summarize_images(self, images):
        messages = []
        for image in images:
            prompt_template = f"""
            Describe one or many images in detail. These images are from a research paper named {self.title}. The provided caption is: "{image["caption"]}".
            Be specific about graphs, such as bar plots.
            Respond only with the describe in short in one paragraph, no additionnal comment.
            Do not start your message by saying "Here is a describe" or anything like that.
            """

            messages.append([("user", [
                    {"type": "text", "text": prompt_template}
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                    for img in image["base64"]
                ])])

        image_summaries = []
        for message in messages:
            prompt = ChatPromptTemplate.from_messages(message)
            chain = prompt | self.multimodal_llm.model | StrOutputParser()
            image_summaries.append(chain.invoke({}))

        # image_summaries = ['The model architecture diagram shows two parallel pathways: an encoder on the left and a decoder on the right. The encoder begins with "Inputs" flowing into "Input Embedding" with added "Positional Encoding," then through a stack of "Nx" identical layers, each containing "Multi-Head Attention" followed by "Add & Norm" and "Feed Forward" followed by "Add & Norm," with skip connections around the attention and feed forward layers. The decoder starts with "Outputs (shifted right)" going into "Output Embedding" with added "Positional Encoding," followed by a similar stack of "Nx" layers, the first containing "Masked Multi-Head Attention" followed by "Add & Norm," then "Multi-Head Attention" followed by "Add & Norm," and finally "Feed Forward" followed by "Add & Norm" with skip connections, leading to a "Linear" layer and a "Softmax" layer to produce "Output Probabilities."',
        # 'The image shows two diagrams side-by-side. The left diagram illustrates "Scaled Dot-Product Attention" with a vertical flow starting from Q, K, and V inputs undergoing a MatMul operation followed by "Scale," an optional "Mask," then a "SoftMax" function, and another "MatMul" to produce the final output. The right diagram depicts "Multi-Head Attention" where multiple "Scaled Dot-Product Attention" blocks, labeled with "h," run in parallel, each preceded by separate "Linear" transformations of V, K, and Q inputs; the outputs of these attention blocks are then "Concat"enated and passed through a final "Linear" layer.',
        # 'The image displays a visualization of attention weights for the word "making" in an encoder self-attention layer. Two rows of text represent the same sentence: "It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult . <EOS> <pad> <pad> <pad> <pad> <pad> <pad>". The word "making" is highlighted in grey. Colored lines extend from the word "making" to other words in the sentence, indicating the attention weights assigned by different attention heads. The colors of the lines represent different attention heads. For example, the word "difficult" receives attention from multiple heads, indicated by lines of different colors connecting "making" to "difficult".',
        # 'The figure displays two sets of attention visualizations, each representing a different attention head\'s focus within layer 5 of a 6-layer model. The top visualization shows full attentions, where each word in the input sequence is connected to every other word with varying line thicknesses and transparency, indicating the strength of attention. The bottom visualization isolates the attentions originating from the word "its" for two separate attention heads, head 5 and head 6, showing highly focused attention on specific words in the sequence with distinct line weights.',
        # 'The figure displays two attention maps illustrating the relationships between words in a sentence, with the same sentence appearing on both the x and y axes of each map. Each map represents the attention weights of a different attention head. The top map uses green lines to connect words, with line thickness indicating the strength of attention. For instance, "The" strongly attends to "Law", and "are" attends to "missing". The bottom map uses red lines in a similar fashion. In this map, "The" attends to itself, and "should" attends to "be". The end-of-sentence token attends to the padding token in both maps.']

        # image_summaries = ['The images show examples of image inpainting. The first image shows a statue of a man on a horse in front of a building, with sections of the image covered in blue. The second image shows the statue and building without the blue sections, which have been inpainted. The third and fourth image show a building with a pool, also with sections covered in blue in the third image, and inpainted in the fourth. The fifth, sixth and seventh image show a portrait of a woman with sections covered in blue in the fifth image, and inpainted in the sixth and seventh.']

        return image_summaries
    
    def add_doc_to_db(self, original_splits, summaries, id_key):
        if original_splits:
            doc_ids = [str(uuid.uuid4()) for _ in summaries]
            summary_texts = [Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(summaries)]
            self.retriever.vectorstore.add_documents(summary_texts)
            self.retriever.docstore.mset(list(zip(doc_ids, original_splits)))

    def update_db(self, file_path):
        self.file_path = file_path
        texts, images, tables = self.extract_doc(file_path)
        # texts, images, tables = [''], [], []
        self.get_doc_title(texts[0])
        text_summaries = self.summarize_texts(texts)
        table_summaries = self.summarize_tables(tables)
        image_summaries = self.summarize_images(images)

        # The vectorstore to use to index the child chunks
        if self.vector_db != None:
            # self.vector_db._collection.delete(self.vector_db._collection.get()['ids'])
            ids = self.vector_db._collection.get()['ids']
            for i in range(0, len(ids), 100):  # Adjust batch size
                self.vector_db._collection.delete(ids[i : i + 100])
        self.vector_db = Chroma(collection_name="multi_modal_rag", embedding_function=self.embedding.model)

        # The retriever (empty to start)
        id_key = "doc_id"
        self.retriever = MultiVectorRetriever(vectorstore=self.vector_db, docstore=InMemoryStore(), id_key=id_key)

        self.add_doc_to_db(texts, text_summaries, id_key)
        self.add_doc_to_db(tables, table_summaries, id_key)
        self.add_doc_to_db(images, image_summaries, id_key)

    def parse_docs(self, docs):
        """Split base64-encoded images, tables, and texts"""
        texts = []
        tables = []
        images = []

        for doc in docs:
            if 'text' in doc.keys():
                texts.append(doc)
            elif 'html' in doc.keys():
                tables.append(doc)
            else:
                images.append(doc)

        return {"texts": texts, "tables": tables, "images": images}

    def build_prompt(self, kwargs):
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]
        context_text = ""

        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element['text']

        # construct prompt with context (including images)
        prompt_template = f"""
        ### INSTRUCTION
        Answer the question based only on the following context, which can include texts, tables, and the below images.

        ### TEXT CONTEXT
        {context_text}

        ### QUESTION
        {user_question}
        """

        prompt_content = [{"type": "text", "text": prompt_template}]

        if len(docs_by_type["tables"]) > 0:
            for table in docs_by_type["tables"]:
                prompt_content.append({"type": "text", "text": f"Caption: {table['caption']}\nTable:\n{table['html']}"})

        if len(docs_by_type["images"]) > 0:
            for image in docs_by_type["images"]:
                prompt_content.append({"type": "text", "text": f"Caption: {image['caption']}"})
                prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image['base64']}"}})

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    def retrieve_doc(self, state):
        # chain = ({
        #         "context": self.retriever | RunnableLambda(self.parse_docs),
        #         "question": RunnablePassthrough(),
        #     }
        #     | RunnableLambda(self.build_prompt)
        #     | self.multimodal_llm.model
        #     | StrOutputParser()
        # )

        chain_with_sources = {
            "context": self.retriever | RunnableLambda(self.parse_docs),
            "question": RunnablePassthrough(),
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self.build_prompt)
                | self.multimodal_llm.model
            )
        )

        response = chain_with_sources.invoke(state['question'])
        # "What do the authors mean by attention?"
        # "What is the attention mechanism?"
        # "What is multihead attention?"

        # self.memory.save_context({'question': state['question'] + '\n ### From document: ' + self.file_path}, {'response': response['response'].content})

        response['document'] = self.file_path
        if 'texts' in response['context']:
            response['context']['texts'] = [{'page_number': text['page_number'], 
                                             'text': text['text'][:100] + '\n...\n' + text['text'][-100:]} 
                                             for text in response['context']['texts']]
        if 'tables' in response['context']['tables']:
            for i in response['context']['tables']:
                del i['html']
        if 'images' in response['context']['images']:
            for i in response['context']['images']:
                del i['base64']

        return {"history": [response['response']], "last_response_refs": response}