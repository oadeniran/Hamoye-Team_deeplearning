from sklearn.base import BaseEstimator, TransformerMixin
class encode_columns(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
    
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        cols_to_transform = list(X.columns)
        if self.columns:
            cols_to_transform  = self.columns
        encoding = {
            'Classification_Size' : {
                'XL' : 1,
                'L' : 2,
                'M' : 3,
                'S' : 4},
            'Research_Intensity' : {
                'VH' : 1,
                'HI' : 2,
                'MD' : 3,
                'LO' : 4},
            'Status' : {
                'A' : 1,
                'B' : 2,
                'C' : 3}
            }
        for col in cols_to_transform:
            if col not in list(encoding.keys()):
                val_dict = {v: i for i, v in enumerate(np.unique(X[col]))}
                #print(val_dict)
            else:
                val_dict = encoding[col]
            X[col] = X[col].map(val_dict)
        return X
