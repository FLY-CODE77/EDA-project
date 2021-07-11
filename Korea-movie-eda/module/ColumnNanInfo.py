def ColumnNanInfo(df, column_name, rval=2):
    '''
    df is input dataframe 
    column_name is which columns want to see the nan value
    rval is round value
    '''
    num1 = (df[column_name] == "정보없음").sum()
    num2 = len(df[column_name])
    percent = round((num1/num2) * 100, rval)
    print(f" {column_name}열 값중 정보없음(nan값)의 퍼센트는 {percent} 입니다")
