TEXT=/home/jovyan/one-billion-words
fairseq-preprocess \
    --srcdict /home/jovyan/one-billion-words/1b_word_vocab.txt \
    --only-source \
    --validpref $TEXT/valid.txt \
    --testpref $TEXT/test.txt \
    --destdir $TEXT/data-bin \
    --workers 20