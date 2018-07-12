import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import LabelEncoder

from pandas.api.types import CategoricalDtype
data_dir = "~/kaggle/zillow/"
all_categorical_vars = ['airconditioningtypeid', 'architecturalstyletypeid',
                    'buildingclasstypeid', 'buildingqualitytypeid', 'decktypeid',
                    'heatingorsystemtypeid','pooltypeid10', 'pooltypeid2', 'pooltypeid7',
                     'propertylandusetypeid', 'regionidcity','regionidcounty', 'regionidneighborhood', 'regionidzip', 'storytypeid',
                    'typeconstructiontypeid', 'fips', 'propertyzoningdesc',
                     'propertycountylandusecode', 'rawcensustractandblock',
                     'censustractandblock', 'assessmentyear']
good_num_vars = ['bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
                 'calculatedfinishedsquarefeet', 'finishedsquarefeet12',
                 'fullbathcnt', 'landtaxvaluedollarcnt', 'latitude',
                 'longitude', 'lotsizesquarefeet', 'roomcnt',
                 'structuretaxvaluedollarcnt', 'taxamount', 'taxvaluedollarcnt',
                 'yearbuilt']
best_num_vars = ['taxamount', 'bedroomcnt','calculatedfinishedsquarefeet']
log_num_vars = ['lotsizesquarefeet']

num_vars_room=['bathroomcnt',"calculatedfinishedsquarefeet", 'calculatedbathnbr',
               "bedroomcnt", "roomcnt"]
num_vars_size=["calculatedfinishedsquarefeet", "finishedsquarefeet12",
               "lotsizesquarefeet"]
num_vars_tax=['structuretaxvaluedollarcnt',"taxamount",
                'taxvaluedollarcnt', 'landtaxvaluedollarcnt']
num_vars_other=['latitude', 'longitude','yearbuilt']
good_cat_vars = ['fips','propertycountylandusecode',
                 'propertylandusetypeid']
ok_cat_vars = ['censustractandblock', 'fips', 'propertycountylandusecode',
                 'propertylandusetypeid', 'rawcensustractandblock',
                 'regionidcity', 'regionidzip']

good_feats = good_num_vars + good_cat_vars

cat_fix = {'fips': {'id_map':{6037:'Los Angeles County',
                              6059:'Orange County',
                              6111:'Ventura County',
                              0:'Not Reported'}},
            'regionidcounty': {'id_map':{3101:'Los Angeles County',
                                          1286:'Orange County',
                                          2061:'Ventura County',
                                          0:'Not Reported'}}}

def load_test_data(zillow_data_dir = data_dir):
    samp = pd.read_csv(zillow_data_dir+'sample_submission.csv', low_memory = False)
    feats = pd.read_csv(zillow_data_dir+'properties_2017.csv', low_memory = False)
    cat_encoders={}
    for cat_var in all_categorical_vars:
        try:
            feats[cat_var] = feats[cat_var].fillna(np.min(feats[cat_var])-10)
        except TypeError:
            feats[cat_var] = feats[cat_var].fillna("not reported")
        cat_encoders[cat_var]=LabelEncoder()
        cat_encoders[cat_var].fit(feats[cat_var].values)
        feats[cat_var] = cat_encoders[cat_var].transform(feats[cat_var])#.astype(np.int32)

    test_df = pd.merge(samp[['ParcelId']],
                       feats.rename(columns = {'parcelid': 'ParcelId'}),
                                            how = 'left', on = 'ParcelId')
    return test_df

def load_train_data(data_dir = data_dir,fill_and_enc_cats = False):
    '''
    loads csv zillow data feature and logerror training labels from csvs,
    joinings on property id, merges 2016, 2017 data into one year
    also parces month, year from date column
    also standardizes longitude/latitude numbers

    returns merged train dataframe
    '''
    cat_encoders={}
    train16 = pd.read_csv(data_dir + 'train_2016_v2.csv', parse_dates=["transactiondate"])
    properties16 = pd.read_csv(data_dir + 'properties_2016.csv')
    merged16 = pd.merge(train16,properties16,on="parcelid",how="left")
    train17 = pd.read_csv(data_dir + 'train_2017.csv', parse_dates=["transactiondate"])
    properties17 = pd.read_csv(data_dir + 'properties_2017.csv')
    merged17 = pd.merge(train17,properties17,on="parcelid",how="left")
    merged = pd.concat([merged16,merged17])
    merged['latitude'] = merged['latitude']/1e6
    merged['longitude'] = merged['longitude']/1e6
    merged["transaction_year"] = merged["transactiondate"].dt.year.astype('int')
    merged["transaction_month"] = merged["transactiondate"].dt.month.astype('int')
    merged['month_year'] = merged['transaction_month'].astype(str)+"_"+merged['transaction_year'].astype(str)
    merged.columns = map(str.lower, merged.columns)
    if fill_and_enc_cats:
        for cat_var in all_categorical_vars:
            try:
                merged[cat_var] = merged[cat_var].fillna(np.min(merged[cat_var])-10)
            except TypeError:
                merged[cat_var] = merged[cat_var].fillna("not reported")
            cat_encoders[cat_var]=LabelEncoder()
            cat_encoders[cat_var].fit(merged[cat_var].values)
            merged[cat_var] = cat_encoders[cat_var].transform(merged[cat_var])#.astype(np.int32)

    print("Shape Of Loaded Merged Zillow Data: ", merged.shape)
    del properties16, properties17, train16, train17
    if fill_and_enc_cats:
        return merged,cat_encoders
    else:
        return merged

def df_percentile_bound(in_df,lower,upper,col_name = 'logerror',get_outliers=False):
    '''
    gets lower, upper percentile values for df column, splits df into bounded
    and outlier dataframes

    returns bound_df, outlier_df, upper lower limits
    '''
    llimit, ulimit = np.percentile(in_df[col_name].values,[lower,upper])
    bound_df=in_df[(in_df[col_name]>llimit) & (in_df[col_name]<ulimit)]
    outlier_df=in_df[(in_df[col_name]<llimit) | (in_df[col_name]>ulimit)]
    if not get_outliers:
        return bound_df
    else:
        return bound_df, outlier_df,[llimit, ulimit]


def cat_float_to_int(in_series):
    in_series.cat.categories = in_series.cat.categories.astype(int)
    return in_series

def calc_iqr(in_num_series):
    feat_iqr = scipy.stats.iqr(in_num_series)
    return [np.median(in_num_series)-feat_iqr,np.median(in_num_series)+feat_iqr]

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()
#import SeabornFig2Grid as sfg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

class snsFig2Grid():
# from https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
