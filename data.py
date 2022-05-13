import pandas as pd

def get_data():
    # get data as dataframe
    url='https://drive.google.com/file/d/1Y2-FuthGBfCqKUhCyWTIt0w8danWx6vm/view?usp=sharing'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url)

    df.columns = ['fLength',
                'fWidth',
                'fSize',
                'fConc',
                'fConc1', 
                'fAsym',
                'fM3Long',
                'fM3Trans',
                'fAlpha',
                'fDist',
                'classified']

    # balance data
    df = df.groupby('classified')
    df = df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True))

    test = df.sample(frac=0.3)
    train = df.copy().drop(test.index)
    
    new_idx = [t[1] for t in train.index]
    train = train.set_index(pd.Series(new_idx))

    new_idx = [t[1] for t in test.index]
    test = test.set_index(pd.Series(new_idx))

    x_train = train.copy()
    y_train = x_train.pop('classified')

    x_test = test.copy()
    y_test = x_test.pop('classified')

    return (x_train,y_train,x_test,y_test)

