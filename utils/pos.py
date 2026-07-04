import torch
import spacy_alignments as tokenizations

TAGS_MAPPING_VERB_NOUN = {
    "VERB": 1,
    "NOUN": 2,
}
TAGS_CONTEXT_MAPPING = {
    "ADJ": 1,
    "ADP": 2,
    "ADV": 3,
}


def get_aligned_pos(sentence, nlp, bert_tokenizer, verbose=False):
    SPECIAL_WORD = "[SPECIAL]"
    # tokenize the sentence and get the POS tags
    # nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    tags = []
    text = []
    for token in doc:
        tags.append(token.pos_)
        text.append(token.text)

    # get the tokenization from bert models
    # bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_tokens = bert_tokenizer.tokenize(
        sentence,
    )
    # get alignment information
    a2b, b2a = tokenizations.get_alignments(text, bert_tokens)
    if verbose:
        for i in range(len(text)):
            print(text[i])
            for j in a2b[i]:
                print("    ", bert_tokens[j])
    # get the POS tags for the bert tokens
    bert_tags = []
    # alwarys start with a CLS special token
    bert_tags.append(SPECIAL_WORD)
    for i in range(len(bert_tokens)):
        bert_tags.append(tags[b2a[i][0]])
    # Align with bert special tokens
    for i in range(40 - len(bert_tags)):
        bert_tags.append(SPECIAL_WORD)
    # Generate the pos tags for verb and noun
    verb_noun_tags = []
    for item in bert_tags:
        if item in TAGS_MAPPING_VERB_NOUN.keys():
            verb_noun_tags.append(TAGS_MAPPING_VERB_NOUN[item])
        elif item == SPECIAL_WORD:
            verb_noun_tags.append(-100)
        else:
            verb_noun_tags.append(0)
    # Generate the context tags
    context_tags = []
    for item in bert_tags:
        if item in TAGS_CONTEXT_MAPPING.keys():
            context_tags.append(TAGS_CONTEXT_MAPPING[item])
        elif item == SPECIAL_WORD:
            context_tags.append(-100)
        else:
            context_tags.append(0)
    # Generate the masks for crf
    mask_verb_noun = []
    for item in verb_noun_tags:
        if item == -100:
            mask_verb_noun.append(0)
        else:
            mask_verb_noun.append(1)
    mask_context = []
    for item in context_tags:
        if item == -100:
            mask_context.append(0)
        else:
            mask_context.append(1)

    return verb_noun_tags, context_tags, mask_verb_noun, mask_context


def compose_pos_from_text_list(text:list, nlp, bert_tokenizer, verbose=False):
    verb_noun = []
    context = []
    verb_noun_mask = []
    context_mask = []
    for item in text:
        verb_noun_tags, context_tags, mask_verb_noun, mask_context = get_aligned_pos(item, nlp, bert_tokenizer, verbose)
        verb_noun.append(verb_noun_tags)
        context.append(context_tags)
        verb_noun_mask.append(mask_verb_noun)
        context_mask.append(mask_context)
    return torch.tensor(verb_noun), torch.tensor(context), torch.tensor(verb_noun_mask), torch.tensor(context_mask)
         
