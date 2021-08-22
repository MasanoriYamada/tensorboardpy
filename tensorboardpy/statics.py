import pandas as pd


def select_steps(df: pd.DataFrame, metric: str, rule: str) -> pd.DataFrame:
    assert metric in set(df['metric'].unique())
    assert rule in {'max', 'min', 'last'}, f'rule is string: max, min, last, but actual value:{rule}'
    hparams = list(set(df.keys()) - {'metric', 'value', 'step'})
    if rule == 'max':
        ddf = df.loc[df['metric'] == metric].groupby(hparams).max()
    elif rule == 'min':
        ddf = df.loc[df['metric'] == metric].groupby(hparams).min()
    elif rule == 'last':
        ddf = df.loc[df['metric'] == metric].groupby(hparams).last()
    # select target step each hparams
    hparams_step = hparams + ['step']
    ddf = ddf.reset_index().set_index(hparams_step)
    dddf = df.set_index(hparams_step)
    ret = []
    dddf = dddf.sort_index()  # for performance
    for key in ddf.index:
        ret.append(dddf.loc[key])
    dddf = pd.concat(ret, axis=0)
    # metric value to colomn name  ここでバグが発生している 全部同じ値になっている
    dddf = dddf.reset_index()
    dddf = pd.pivot_table(data=dddf, index=list(set(df.keys()) - {'metric', 'value'}), columns='metric', values='value').reset_index()
    dddf.columns.name = None
    return dddf
