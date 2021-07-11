
def WordCount(df, column_name, delval="정보없음", column_name1="영화명",rank=20):
    '''
    df is input dataframe
    column_name is which columns you want to drop
    delval is value that you want to delete
    column_name1 is which columns you want to see Rank word data
    rank Rank you want to see 
    '''
    # import,load stopword
    import nltk
    from konlpy.tag import Okt; t = Okt()
    import pandas as pd
    import seaborn as sns

    from module.DeleteVal import DeleteVal
    from module.WordCount import WordCount
    import matplotlib.pylab as plt
    
    import os
    file_path = os.getcwd()
    file_path 
    file_path = file_path +"/" + "data" + "/" +"한국어불용어100.txt"
    print(file_path)
    stop_words = pd.read_csv(file_path, sep = "\t", engine='python')["stop"]
    stop_words = list(stop_words)
    
    # DeleteVal from columns and filter by columns1
    word_column = DeleteVal(df, column_name)[column_name1]
    
    # text data
    present_text = ''
    for each_line in word_column:
        present_text = present_text + each_line + '\n'
    
    # okt and nltk and make Rank data
    tokens_ko = t.nouns(present_text)
    tokens_ko = [each_word for each_word in tokens_ko if each_word not in stop_words]
    ko = nltk.Text(tokens_ko)
    Rank_data = ko.vocab().most_common(rank)
    
    # graph
    x, y = dict(Rank_data).keys(), dict(Rank_data).values()
    plt.figure(figsize=(15,8))
    plt.title(f"자주 쓰인 단어 {rank}개", fontsize=20)
    sns.barplot(list(x), list(y), palette="Dark2")
    plt.savefig("./images/wordcount.png", facecolor='#ffffff')
    plt.show()
    
    # return Rank_data for next use 
    return Rank_data
