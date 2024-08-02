from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.data.dictionary import Dictionary
from fairseq.fairseq_cli.eval_lm import eval_lm
fairseq_model = TransformerLanguageModel.from_pretrained('/base-vol-2/fairseq/models/adaptive_lm_wiki103.v2/', 'model.pt')
source_dictionary = Dictionary.load('/base-vol-2/fairseq/models/adaptive_lm_wiki103.v2/dict.txt')


eval_lm(fairseq_model, source_dictionary,)