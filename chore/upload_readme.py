"""
Make sure standalone code for running inference on uploaded models work
"""


if __name__ == '__main__':
    def binary_bert_instr():
        from zeroshot_classifier.models import BinaryBertCrossEncoder
        model = BinaryBertCrossEncoder(model_name='claritylab/zero-shot-vanilla-binary-bert')

        text = "I'd like to have this track onto my Classical Relaxations playlist."
        labels = [
            'Add To Playlist', 'Book Restaurant', 'Get Weather', 'Play Music', 'Rate Book', 'Search Creative Work',
            'Search Screening Event'
        ]
        query = [[text, lb] for lb in labels]
        logits = model.predict(query, apply_softmax=True)
        print(logits)
    # binary_bert_instr()

    def bi_encoder_instr():
        from sentence_transformers import SentenceTransformer, util as sbert_util

        model = SentenceTransformer(model_name_or_path='claritylab/zero-shot-implicit-bi-encoder')

        text = "I'd like to have this track onto my Classical Relaxations playlist."
        labels = [
            'Add To Playlist', 'Book Restaurant', 'Get Weather', 'Play Music', 'Rate Book', 'Search Creative Work',
            'Search Screening Event'
        ]
        text_embed = model.encode(text)
        label_embeds = model.encode(labels)

        scores = [sbert_util.cos_sim(text_embed, lb_embed).item() for lb_embed in label_embeds]
        print(scores)
    # bi_encoder_instr()

    def gpt2_instr():
        import torch

        from zeroshot_classifier.models import ZsGPT2Tokenizer, ZsGPT2LMHeadModel

        model_name = 'claritylab/zero-shot-vanilla-gpt2'
        model = ZsGPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = ZsGPT2Tokenizer.from_pretrained(model_name, form='vanilla')

        model.config.pad_token_id = model.config.eos_token_id
        model.eval()

        # 'dataset_name` just so that it passes, irrelevant
        text = "I'd like to have this track onto my Classical Relaxations playlist."
        labels = [
            'Add To Playlist', 'Book Restaurant', 'Get Weather', 'Play Music', 'Rate Book', 'Search Creative Work',
            'Search Screening Event'
        ]
        inputs = tokenizer(
            dict(text=text, label_options=labels), mode='inference-sample'
        )

        inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_length=128)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        print(decoded)
    gpt2_instr()
