# Demand extraction base on conversation

## Aim:
.The demand trend detect of client conversation with employees  <br> .Demand extraction and apply for future agents reminder under task handle circumstance or automatically Robot Q&A

## word_cut-extract_judge code method
-> Data preprocessing <br>
-> Built StopWord dict and AddWord dict <br>
-> Original Text cut words into list (base on two dicts) <br>
-> Keyword Extraction (base on tfidf or TextRank) <br>
-> Rule dictionary built (like {"Tag":[k1,k2,k3]} currently by manual) <br>
-> Sentences Text import and Judge module apply <br>

## Stage 2 been Done
.N_gram method for high probability rule dictionary update <br>
.N_gram to N_follow judge fix match to dynamic match, aiming to provide a wider demand construction <br>

## Future Plan and issue
.Low Recall on all terms especially on Covid term <br>
.Futrue Plan: Add more realted terms to fill the gap, About short term trend. considering single words bags to detect. As for long term trend, considering realted words bags for detection. <br>

## Future Plan
Automatic Model to detect high probability words match
