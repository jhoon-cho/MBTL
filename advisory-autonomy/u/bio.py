from . import *

Chroms = sorted(['chr' + x for x in ([str(i) for i in range(1, 23)] + ['X', 'Y'])])
Bed_header = ['chr', 'start', 'end']
Bed_header_name = Bed_header + ['name']
Fasta_header = ['name', 'sequence']
Vcf_header = ['chr', 'pos', 'name', 'ref', 'alt']
Chroms = sorted(['chr' + x for x in ([str(i) for i in range(1, 23)] + ['X', 'Y'])])
Bases = ['A', 'C', 'G', 'T', 'N']

def ks_test(y_true, y_pred):
    return sp.stats.ks_2samp(y_pred[y_true == 0], y_pred[y_true == 1])

def bh_correction(pvals):
    pvals_array = np.array(pvals)
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector
    stats = importr('stats')
    qvals = np.array(stats.p_adjust(FloatVector(pvals_array.reshape((-1,))), method='BH'))
    if type(pvals) == pd.DataFrame:
        return pd.DataFrame(qvals.reshape(pvals_array.shape), index=pvals.index, columns=pvals.columns)
    elif type(pvals) == pd.Series:
        return pd.Series(qvals, index=pvals.index, name=pvals.name)
    return qvals

def substitute(seq, pos, ref, alt):
    pos_end = pos + len(ref)
    assert seq[pos : pos_end] == ref, 'sequence %s and ref %s do not match.' % (seq[pos : pos_end], ref)
    return seq[:pos] + alt + seq[pos_end:]

def resize_length(df, new_length, start='start', end='end', new_start='start', new_end='end'):
    '''
    resize the sequence around the center. Center is center for odd length, right of center for even length.
    '''
    convert_columns_to_numeric(df, [start, end])
    middle = (df[start] + df[end]) // 2
    front_length = new_length // 2
    back_length = new_length - front_length
    df[new_start] = middle - front_length
    df[new_end] = middle + back_length
    return df

def append_bed_columns(df, column):
    df[Bed_header] = df[column].str.extract('(\w+):(\d+)-(\d+)', expand=True)
    for column in 'start', 'end':
        df[column] = df[column].astype(int)
    return df

def append_bed_summary_column(df, column):
    df[column] = df.apply(lambda row: '%s:%s-%s' % (row['chr'], row['start'], row['end']), axis=1)
    return df

def convert_columns_to_numeric(df, columns):
    df[columns] = df[columns].apply(pd.to_numeric)
    return df

class NewPath(Path):
    def load_fasta(self, has_name=False):
        data = []
        with open(self, 'r') as f:
            line = f.readline()
            while line:
                name = line.rstrip()[1:]
                if has_name:
                    name = name.split('::')[0]
                data.append((name, f.readline().rstrip()))
                line = f.readline()
        return pd.DataFrame(data, columns=Fasta_header)
    
    def save_fasta(self, df, columns=Fasta_header):
        with open(self, 'w') as f:
            for i, row in df.iterrows():
                f.write('>%s\n%s\n' % tuple(row[columns]))

    def read_header(self, sep=None):
        with open(self, 'r') as f:
            return f.readline().rstrip().split(sep)

    def load_bed(self):
        if len(self.read_header(sep='\t')) == 3:
            names = Bed_header
        else:
            names = Bed_header_name
        return pd.read_csv(self, sep='\t', names=names)

    def save_bed(self, df, columns=Bed_header):
        df[columns].to_csv(self, sep='\t', index=False, header=False)

    def load_narrowpeak(self):
        return pd.read_csv(self, sep='\t', header=None, names=[
            'chrom', 'chromStart', 'chromEnd',
            'name', 'score', 'strand',
            'signalValue', 'pValue', 'qValue', 'peak'
        ])

    def load_vcf(self):
        return pd.read_csv(self, sep='\t', names=Vcf_header, keep_default_na=False)

    def save_vcf(self, df, columns=Vcf_header):
        df[Vcf_header].to_csv(self, sep='\t', index=False, header=False)

Path = NewPath

from sklearn import metrics
def get_reg_correlations(y_true, y_pred, extended=False):
    correlators = (sp.stats.spearmanr, sp.stats.pearsonr, sp.stats.kendalltau)
    labels = ('Spearman', 'Pearson', 'KendalTau')

    correlations = pd.DataFrame(np.nan, index=labels, columns=['value', 'pvalue'])
    correlations.index.name = 'correlator'
    correlations.columns.name = 'category'
    if len(y_true) == len(y_pred) == 0:
        return correlations
    for correlator, label in zip(correlators, labels):
        correlations.loc[label] = correlator(y_true, y_pred)
    if not extended:
        return correlations
    top_25_cutoff = np.percentile(y_true, 75)
    top_25_indices = y_true > top_25_cutoff
    y_true_top_25 = y_true[top_25_indices]
    y_pred_top_25 = y_pred[top_25_indices]
    for correlator, label in zip(correlators, labels):
        correlation = 0
        if len(y_true_top_25) > 0:
            correlations.loc[label + '_top_25'] = correlator(y_true_top_25, y_pred_top_25)
    y_true_binned = bin_percentiles(y_true)
    y_pred_binned = bin_percentiles(y_pred)
    correlations.loc['Spearman_binned'] = sp.stats.spearmanr(y_true_binned, y_pred_binned)
    return correlations

def get_cls_correlations(y_true, y_pred, return_p_value=False):
    correlations = pd.DataFrame(np.nan, index=['AUROC', 'AUPRC'], columns=['value'])
    correlations.index.name = 'correlator'
    correlations.columns.name = 'category'
    if len(y_true) == len(y_pred) == 0:
        return correlations
    correlations.loc['AUROC'] = metrics.roc_auc_score(y_true, y_pred)
    correlations.loc['AUPRC'] = metrics.average_precision_score(y_true, y_pred)
    return correlations

from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNetCV, LinearRegression
from sklearn.base import clone

class VotingRegressor:
    def __init__(self, regressors, clone_regressors=True):
        if clone_regressors:
            self.estimators_ = [clone(reg) for reg in regressors]
        else:
            self.estimators_ = regressors

    def fit(self, X_train, y_train):
        failed_regs = []
        for reg in self.estimators_:
            try:
                reg.fit(X_train, y_train)
            except (np.linalg.linalg.LinAlgError, ValueError) as e:
                print(str(e))
                failed_regs.append(reg)
                continue
        for reg in failed_regs:
            self.estimators_.remove(reg)

    def predict(self, X_test):
        preds_test = []
        for reg in self.estimators_:
            prediction = reg.predict(X_test)
            preds_test.append(prediction)
        preds_test = np.array(preds_test)
        return np.mean(preds_test, axis=0)

def sk_get_voting_classifier(classifiers, voting='soft'):
    return VotingClassifier([(cls.__class__.__name__ + '_' + str(i), cls) for i, cls in enumerate(classifiers)], voting=voting)

def sk_get_voting_regressor(regressors):
    return VotingRegressor(regressors)

sk_classifiers = [
    RandomForestClassifier(n_estimators=1000, max_features='sqrt', n_jobs=-1),
    ExtraTreesClassifier(n_estimators=1000, max_features='sqrt', n_jobs=-1),
]
sk_quick_classifiers = [
    ExtraTreesClassifier(n_estimators=1000, max_features='sqrt', n_jobs=-1),
]
sk_regressors = [
    RandomForestRegressor(n_estimators=1000, max_features='sqrt', n_jobs=-1),
    ExtraTreesRegressor(n_estimators=1000, max_features='sqrt', n_jobs=-1),
    GradientBoostingRegressor(n_estimators=1000),
    BayesianRidge(),
    ElasticNetCV(),
]
sk_quick_regressors = [
    ExtraTreesRegressor(n_estimators=1000, max_features='sqrt', n_jobs=-1)
]
