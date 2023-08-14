# nna-demo

1. Install the packages using command: **pip install -r regularvicuna_requirments.txt**
2. Download the model(ggml-vic13b-q8_0.bin) from https://huggingface.co/eachadea/ggml-vicuna-13b-1.1/tree/main
3. Place the model in /models directory.
4. Run **python embeddingindex.py** command to create the indexes for the chunks created.
5. Run **streamlit run vicuna.py** to run the application in streamlit and for the UI.
