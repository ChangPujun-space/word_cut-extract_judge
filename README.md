# Demand extraction base on conversation
## Aim:
.The demand trend detect of client conversation with employees  <br> .Demand extraction and apply for future agents reminder under task handle circumstance or automatically Robot Q&A
## word_cut-extract_judge code method
-> Data preprocessing <br>
-> Built StopWord dict and AddWord dict <br>
-> Word cut into list (base on two dicts) <br>
-> Keyword Extraction (base on tfidf or TextRank) <br>
-> Rule dictionary built (like {"Tag":[k1,k2,k3]}) <br>
-> Sentences import and Judge module apply <br>
## Future Plan
.N_gram method for high probability rule dictionary update <br>
.The Rule for between Tags in Rule Dictionary
