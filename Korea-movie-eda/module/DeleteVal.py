def DeleteVal(df, column_name, delval="정보없음"):
    '''
    df is input dataframe
    column_name is which columns you want to drop
    delval is value that you want to delete
    '''
    df = df[df[column_name] != delval]
    return df
