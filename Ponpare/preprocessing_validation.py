import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
import sys
import cPickle as pickle
from datetime import date
import calendar
import scipy.io as spi
import scipy.sparse as sps
from sklearn.feature_extraction import DictVectorizer


def create_LabelEncoded_files(week_ID) :
    ''' Apply LabelEncoding to the data to create:
         A file with test coupon information (and LabelEncoding of categorical variables)
        A file which aggregates coupon_detail and user information (and LabelEncoding of categorical variables)
        These files will be used in the similarity_distance.py script
    '''

    print 'Create Label Encoded files'

    def get_unix_time(row):

        '''Convert to unix time. Neglect time of the day
        '''
        row = row.split(' ')
        row = row[0].split('-')
        y,m,d = int(row[0]), int(row[1]), int(row[2])

        return calendar.timegm(date(y,m,d).timetuple())

    def get_day_of_week(row):
        '''Convert to unix time. Neglect time of the day
        '''
        row = row.split(' ')
        row = row[0].split('-')
        y,m,d = int(row[0]), int(row[1]), int(row[2])

        return date(y,m,d).weekday()

    #Read all the input data
    cpdtr = pd.read_csv('../Data/Validation/%s/coupon_detail_train_validation_%s.csv' % (week_ID, week_ID))
    cpltr = pd.read_csv('../Data/Validation/%s/coupon_list_train_validation_%s.csv' % (week_ID, week_ID))
    cplte = pd.read_csv('../Data/Validation/%s/coupon_list_test_validation_%s.csv' % (week_ID, week_ID))
    ulist = pd.read_csv('../Data/Validation/%s/user_list_validation_%s.csv' % (week_ID, week_ID))
    ulist['REG_DATE_UNIX'] = ulist['REG_DATE'].apply(get_unix_time)
    cplte['DISPFROM_day'] = cplte['DISPFROM'].apply(get_day_of_week)
    cpltr['DISPFROM_day'] = cpltr['DISPFROM'].apply(get_day_of_week)
    cplte['DISPEND_day'] = cplte['DISPEND'].apply(get_day_of_week)
    cpltr['DISPEND_day'] = cpltr['DISPEND'].apply(get_day_of_week)

    # List of features we will play with
    list_col = ['COUPON_ID_hash','USER_ID_hash', 'GENRE_NAME', 'large_area_name', 'small_area_name', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE',
       'DISPFROM', 'DISPEND','DISPFROM_day', 'DISPEND_day', 'DISPPERIOD', 'VALIDFROM', 'VALIDEND',
       'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE',
       'USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI',
       'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
       'USABLE_DATE_BEFORE_HOLIDAY', 'ITEM_COUNT', 'AGE', 'SEX_ID', 'REG_DATE_UNIX']

    #making of the train set
    train = pd.merge(cpdtr, cpltr)
    train = pd.merge(train, ulist, left_on = 'USER_ID_hash', right_on = 'USER_ID_hash')
    train = train[list_col]

    # Format the test set as the train set
    cplte['USER_ID_hash'] = np.array(['dummyuser']*len(cplte))
    for col in ['ITEM_COUNT', 'AGE', 'SEX_ID', 'REG_DATE_UNIX'] :
        cplte[col] = 0

    #Combine test and train to apply LabelEncoding
    cpchar = cplte[list_col]
    train = pd.concat([train, cpchar])

    # Use a sklearn_pandas mapper for Label Encoding
    list_mapper      = []
    # Store LabelEncoded col names in new list
    for feat in list_col :
        if feat in ['GENRE_NAME', 'large_area_name', 'small_area_name'] :
            list_mapper.append((feat, preprocessing.LabelEncoder()))
        else :
            list_mapper.append((feat, None))
    mapper = DataFrameMapper(list_mapper)

    # Fit LabelEncoder
    train = mapper.fit_transform(train)
    # Build df of LabelEncoded features
    train = pd.DataFrame(train, index = None, columns = list_col )

    #separate the test from train
    test = train[train['USER_ID_hash']=='dummyuser']
    train = train[train['USER_ID_hash'] !='dummyuser']

    #Save the test data
    test.to_csv('../Data/Validation/%s/coupon_list_test_LE_validation_%s.csv' % (week_ID, week_ID), index = False)
    #Free memory
    del test

    #Save the train data
    train.to_csv('../Data/Validation/%s/coupon_train_aggregated_LE_validation_%s.csv' % (week_ID, week_ID), index = False)
    #Free memory
    del train


def create_LabelBinarized_files(week_ID) :
    ''' Apply LabelBinarizing to the data to create:
         A file with test coupon information (and LabelBinarizing of categorical variables)
        A file which aggregates coupon_detail and user information (and LabelBinarizing of categorical variables)
        A file which aggregates coupon_visit and user information (and LabelBinarizing of categorical variables)
        These files will be used in the similarity_distance.py script

        arg : week_ID (str) validation week
    '''

    print 'Create Label Binarized files'

    def get_unix_time(row):
        '''Convert to unix time. Neglect time of the day
        '''
        row = row.split(' ')
        row = row[0].split('-')
        y,m,d = int(row[0]), int(row[1]), int(row[2])
        return calendar.timegm(date(y,m,d).timetuple())

    #read in all the input data
    cpdtr = pd.read_csv('../Data/Validation/%s/coupon_detail_train_validation_%s.csv' % (week_ID, week_ID))
    cpltr = pd.read_csv('../Data/Validation/%s/coupon_list_train_validation_%s.csv' % (week_ID, week_ID))
    cplte = pd.read_csv('../Data/Validation/%s/coupon_list_test_validation_%s.csv' % (week_ID, week_ID))
    ulist = pd.read_csv('../Data/Validation/%s/user_list_validation_%s.csv' % (week_ID, week_ID))
    ulist['REG_DATE_UNIX'] = ulist['REG_DATE'].apply(get_unix_time)

    # List of unbinarized features
    list_col_unbin = ['COUPON_ID_hash','USER_ID_hash', 'GENRE_NAME', 'large_area_name', 'small_area_name', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE',
       'DISPFROM', 'DISPEND', 'DISPPERIOD', 'VALIDFROM', 'VALIDEND',
       'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE',
       'USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI',
       'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
       'USABLE_DATE_BEFORE_HOLIDAY', 'ITEM_COUNT', 'AGE', 'SEX_ID', 'REG_DATE_UNIX']

    #making of the train set
    train = pd.merge(cpdtr, cpltr)
    train = pd.merge(train, ulist, left_on = 'USER_ID_hash', right_on = 'USER_ID_hash')
    train = train[list_col_unbin]

    # Format the test set as the train set
    cplte['USER_ID_hash'] = np.array(['dummyuser']*len(cplte))
    for col in ['ITEM_COUNT', 'AGE', 'SEX_ID', 'REG_DATE_UNIX'] :
        cplte[col] = 0
    #Then combine test and train
    cpchar = cplte[list_col_unbin]
    train = pd.concat([train, cpchar])

    # Binarize features now
    list_to_binarize = ['GENRE_NAME', 'large_area_name', 'small_area_name']
    # After binarisation, we obtain more features. We store the name of those features in d_bin
    d_bin = {}
    for feat in list_to_binarize:
        if feat == 'GENRE_NAME' :
            cardinal = sorted(set(train[feat].values))
            d_bin['GENRE_NAME'] = [feat + '_' + str(i) for i in cardinal]
        if feat == 'large_area_name' :
            cardinal = sorted(set(train[feat].values))
            d_bin['large_area_name'] = [feat + '_' + str(i) for i in cardinal]
        if feat == 'small_area_name' :
            cardinal = sorted(set(train[feat].values))
            d_bin['small_area_name'] = [feat + '_' + str(i) for i in cardinal]

    # Use a sklearn_pandas mapper for binarization
    list_mapper      = []
    # Store binaried col names in new list
    list_col_bin = []
    for feat in list_col_unbin :
        if feat in list_to_binarize :
            list_col_bin += d_bin[feat]
            list_mapper.append((feat, preprocessing.LabelBinarizer()))
        else :
            list_col_bin.append(feat)
            list_mapper.append((feat, None))
    mapper = DataFrameMapper(list_mapper)

    # Fit binarizer of full matrix and save
    train = mapper.fit_transform(train)
    # Incorporate binarized feature in train
    train = pd.DataFrame(train, index = None, columns = list_col_bin )

    #separate the test from train
    test = train[train['USER_ID_hash']=='dummyuser']
    train = train[train['USER_ID_hash'] !='dummyuser']

    #Save the test data
    test.to_csv('../Data/Validation/%s/coupon_list_test_LB_validation_%s.csv' % (week_ID, week_ID), index = False)
    #Free memory
    del test

    #Save the train data
    train.to_csv('../Data/Validation/%s/coupon_train_aggregated_LB_validation_%s.csv' % (week_ID, week_ID), index = False)
    #Free memory
    del train

    #Load visit data frame in chunks because it is too large
    for index, cpvtr in enumerate(pd.read_csv('../Data/Validation/%s/coupon_visit_train_validation_%s.csv' % (week_ID, week_ID), chunksize=100000)) :
        sys.stdout.write('\rProcessing row ' + str(index*100000)+' to row '+str((index+1)*100000))
        sys.stdout.flush()
        cpvtr = cpvtr[cpvtr['PURCHASE_FLG']!=1][['VIEW_COUPON_ID_hash','USER_ID_hash']]
        trainv = pd.merge(cpvtr, cpltr, left_on = 'VIEW_COUPON_ID_hash', right_on = 'COUPON_ID_hash')
        trainv = pd.merge(trainv, ulist, left_on = 'USER_ID_hash', right_on = 'USER_ID_hash')
        trainv['ITEM_COUNT'] = 0
        trainv = trainv[list_col_unbin]

        #Binarize
        trainv = mapper.transform(trainv)
        trainv = pd.DataFrame(trainv, index = None, columns = list_col_bin )

        # Add trainv to trainvisit
        if index == 0:
            with open('../Data/Validation/%s/coupon_trainv_aggregated_LB_validation_%s.csv' % (week_ID, week_ID), 'w') as f :
                trainv.to_csv(f, index = False)
        else :
            with open('../Data/Validation/%s/coupon_trainv_aggregated_LB_validation_%s.csv' % (week_ID, week_ID), 'a') as f :
                trainv.to_csv(f, index = False, header=False)

    print


def prepare_similarity_data(var_choice, week_ID):
    '''
    Prepare the dataframes of user and coupon characteristics.
    We build a representation of users and coupons in the same vector space.

    1) For a given coupon, the associated vector is simply the feature vector.

    2)For a given user, we look at his coupon visit and detail history. We then take the mean of all
    the coupons viewed/purchase to obtain the user's representation in the coupon feature space.
    N.B. the contribution of the visit history is weighted by a factor called couponVisitFactor
    which we set = 0.01

    These representations are stored in a pandas DataFrame
    (uchar for users and test for the test coupons)

    args : var_choice (str) Selects desired feature engineering
    For now, a single choice of feature engineering to use has been implemented.
    It is called '1'.
    week_ID (str) validation week
    '''

    print 'Create files for similarity recommendation'

    #Load dictionary which stores the list of users without data
    d_user_list = {}
    with open('../Data/Validation/%s/dict_user_list_validation_%s.pickle' % (week_ID, week_ID), 'r') as fp:
        d_user_list = pickle.load(fp)

    # Load train, test and user list
    test = pd.read_csv('../Data/Validation/%s/coupon_list_test_LB_validation_%s.csv' % (week_ID, week_ID))
    train = pd.read_csv('../Data/Validation/%s/coupon_train_aggregated_LB_validation_%s.csv' % (week_ID, week_ID))
    ulist = pd.read_csv('../Data/Validation/%s/user_list_validation_%s.csv' % (week_ID, week_ID))

    #List of binarized columns
    list_gnr = [col for col in train.columns.values if 'GENRE' in col]
    list_large = [col for col in train.columns.values if 'large' in col]
    list_small = [col for col in train.columns.values if 'small' in col]

    #Utility functions
    def group_user(uc, ucv, couponVisitFactor) :
        ''' group uchar from detail (uc) and from visit (ucv) using pandas groupby.
        The groupby is on USER_ID_hash and we apply mean().
        Users without data (no purchase/ no visit) are assigned the mean of all other users'
        representation
        '''

        # Group uc and ucv (apply mean)
        g = uc.groupby('USER_ID_hash', as_index = False)
        uc = g.mean()
        g = ucv.groupby('USER_ID_hash', as_index = False)
        ucv = g.mean()
        # Multiply ucv by the couponVisitFactor
        for col in ucv.columns[1:]:
            ucv[col] = couponVisitFactor * ucv[col]
        # Aggregate uc and ucv
        uc = uc.append(ucv)
        g = uc.groupby('USER_ID_hash', as_index = False)
        uc = g.mean()
        # Impute uc for users without view and without detail
        list_index_no_view_no_detail = [list(uc['USER_ID_hash'].values).index(el) for el in d_user_list['no_view_no_detail_user']]
        uc_mean = uc.iloc[:,1:].mean(axis = 0).values
        # #For users without any information, predictions based on mean
        uc.iloc[list_index_no_view_no_detail, 1:] = uc_mean
        return uc

    def initialise_uchar(ulist, train):
        '''Initalise user characteristic dataframes (uchar : detail, ucharv : view)
        args : ulist (dataframe) list of users
        train (dataframe) detail dataframe (need its dimensions to initialise uchar)
        '''
        # Create vector of user characteristics
        uchar = pd.DataFrame(ulist['USER_ID_hash'].values, index=None, columns = ['USER_ID_hash'])
        empty_df = pd.DataFrame(np.zeros( (len(uchar), train.shape[1]-2) ), columns = list(train.columns.values[2:]))
        uchar = pd.concat([uchar, empty_df], axis =1)
        ucharv = uchar.copy()
        return uchar, ucharv

    if var_choice == '1' :

        #List of columns to keep for this var_choice
        list_col = ['COUPON_ID_hash', 'USER_ID_hash'] + list_gnr + ['DISCOUNT_PRICE', 'DISPPERIOD']
        list_col+= list_large + list_small + ['VALIDPERIOD_0','VALIDPERIOD_1', 'USABLE_DATE_sum', 'ITEM_COUNT']

        #Feature engineering
        def get_feat(df, name):
            ''' Feature engineering
            args : df (dataframe)
                name (str) type of dataframe to process (test, train and trainv)
            '''
            df['VALIDPERIOD'] = df['VALIDPERIOD'].fillna(-1)
            df['VALIDPERIOD'] = df['VALIDPERIOD']+1
            df.loc[df['VALIDPERIOD']>0, 'VALIDPERIOD'] = 1

            # Binarize VALIDPERIOD here (LabelBinarizer() from scikit does not provide a 2D output when there are only 2 classes)
            df['VALIDPERIOD_0'] = np.zeros(len(df))
            df['VALIDPERIOD_1'] = np.zeros(len(df))
            df.loc[df['VALIDPERIOD']==0, 'VALIDPERIOD_0'] =1
            df.loc[df['VALIDPERIOD']==1, 'VALIDPERIOD_1'] =1
            df = df.drop('VALIDPERIOD', 1)

            if name == 'test' :
                # List of columns USABLE_DATE_XXX
                usable_col = [col for col in df.columns.values if 'USABLE' in col]
                # sets up sum of coupon USABLE_DATEs for training and test dataset
                for col in usable_col :
                    df[col] = df[col].fillna(0)
                    df.loc[df[col]>1, col] = 1
                df['USABLE_DATE_sum'] = df[usable_col].sum(axis=1)
                df['DISCOUNT_PRICE'] = 1./np.log10(df['DISCOUNT_PRICE'])
                df.loc[df['DISPPERIOD']>7, 'DISPPERIOD'] = 7
                df['DISPPERIOD'] = df['DISPPERIOD']/7
                df['USABLE_DATE_sum'] = df['USABLE_DATE_sum']/9

            else :
                df['DISCOUNT_PRICE'] = 1
                df['DISPPERIOD'] = 1
                df['USABLE_DATE_sum'] = 1

            df = df.fillna(1)

            return df[list_col]

        train = get_feat(train, 'train')
        test = get_feat(test, 'test')

        # Create vector of user characteristics
        uchar, ucharv = initialise_uchar(ulist, train)

        # Incorporate the purchase training data from train
        uchar = uchar.append(train.iloc[:,1:])
        # Multiply by item count
        uchar[uchar.columns.values[1:-1]] = uchar[uchar.columns.values[1:-1]].multiply(uchar['ITEM_COUNT'], axis = 0)

        #Incrementally read the trainv file (user view info)
        for index, trainv in enumerate(pd.read_csv('../Data/Validation/%s/coupon_trainv_aggregated_LB_validation_%s.csv' % (week_ID, week_ID), chunksize=100000)) :
            sys.stdout.write('\rProcessing row ' + str(index*100000)+' to row '+str((index+1)*100000))
            sys.stdout.flush()
            trainv = get_feat(trainv, 'trainv')
            ucharv = ucharv.append(trainv.iloc[:,1:])

        print
        #Drop the columns we no longer need
        uchar = uchar.drop('ITEM_COUNT', 1)
        ucharv = ucharv.drop('ITEM_COUNT', 1)
        test = test.drop(['ITEM_COUNT', 'USER_ID_hash'], 1)

        # Define couponVisitFactor and apply groupby to uchar and ucharv
        couponVisitFactor = 0.01
        uchar = group_user(uchar, ucharv, couponVisitFactor)

        # Save data
        uchar.to_csv('../Data/Validation/%s/uchar_var_%s_train_validation_%s.csv' % (week_ID, var_choice, week_ID), index = False)
        test.to_csv('../Data/Validation/%s/test_var_%s_validation_%s.csv' % (week_ID, var_choice, week_ID), index = False)

#########################################
# Below : preprocessing to use LightFM
##########################################

def build_biclass_user_item_mtrx(week_ID):
    ''' Build user item matrix (for test and train datasets)
    (sparse matrix, Mui[u,i] = 1 if user u has purchase item i, -1 if viewed, not purchased)

    arg : week_ID (str) validation week
    '''

    print 'Creating biclass user_item matrix for LightFM'

    #For now, only consider the detail dataset
    cpvtr = pd.read_csv('../Data/Validation/%s/coupon_visit_train_validation_%s.csv' % (week_ID, week_ID))
    cpdtr = pd.read_csv('../Data/Validation/%s/coupon_detail_train_validation_%s.csv' % (week_ID, week_ID))
    cpltr = pd.read_csv('../Data/Validation/%s/coupon_list_train_validation_%s.csv' % (week_ID, week_ID))
    cplte = pd.read_csv('../Data/Validation/%s/coupon_list_test_validation_%s.csv' % (week_ID, week_ID))
    ulist = pd.read_csv('../Data/Validation/%s/user_list_validation_%s.csv' % (week_ID, week_ID))

    #Only consider views without purchase
    cpvtr = cpvtr[cpvtr['PURCHASE_FLG'] == 0]

    #Build a dict with the coupon index in cpltr
    d_ci_tr = {}
    for i in range(len(cpltr)) :
        coupon = cpltr['COUPON_ID_hash'].values[i]
        d_ci_tr[coupon] = i

    #Build a dict with the coupon index in cplte
    d_ci_te = {}
    for i in range(len(cplte)) :
        coupon = cplte['COUPON_ID_hash'].values[i]
        d_ci_te[coupon] = i

    #Build a dict with the user index in ulist
    d_ui = {}
    for i in range(len(ulist)) :
        user       = ulist['USER_ID_hash'].values[i]
        d_ui[user] = i

    #Build the user x item matrices using scipy lil_matrix
    Mui_tr = sps.lil_matrix((len(ulist), len(cpltr)), dtype=np.int8)

    #Fill the Mui_tr train matrix from info in cpvtr
    for i in range(len(cpvtr)) :
        sys.stdout.write('\rProcessing row ' + str(i)+'/ '+str(cpvtr.shape[0]))
        sys.stdout.flush()
        user        = cpvtr['USER_ID_hash'].values[i]
        coupon      = cpvtr['VIEW_COUPON_ID_hash'].values[i]
        # Exception for coupons viewed before they can be purchased
        try :
            ui, ci      = d_ui[user], d_ci_tr[coupon]
            Mui_tr[ui, ci] = -1
        except KeyError :
            pass
    print
    #Now fill Mui_tr with the info from cpdtr
    for i in range(len(cpdtr)) :
        sys.stdout.write('\rProcessing row ' + str(i)+'/ '+str(cpdtr.shape[0]))
        sys.stdout.flush()
        user        = cpdtr['USER_ID_hash'].values[i]
        coupon      = cpdtr['COUPON_ID_hash'].values[i]
        ui, ci      = d_ui[user], d_ci_tr[coupon]
        Mui_tr[ui, ci] = 1
    print

    print
    #Save the matrix in the COO format
    spi.mmwrite('../Data/Validation/%s/biclass_user_item_train_mtrx_%s.mtx' % (week_ID, week_ID), Mui_tr)


def build_user_feature_matrix(week_ID):
    ''' Build user feature matrix
    (feat = AGE, SEX_ID, these feat are then binarized)

    arg : week_ID (str) validation week

    '''

    print 'Creating user_feature matrix for LightFM'

    def age_function(age, age_low =0, age_up = 100):
        '''Function to binarize age in age slices
        '''
        if age_low<= age < age_up :
            return 1
        else :
            return 0

    def format_reg_date(row):
        '''Format reg date to 'year-month'
        '''
        row = row.split(' ')
        row = row[0].split('-')
        reg_date = row[0] #+ row[1]
        return reg_date

    ulist = pd.read_csv('../Data/Validation/%s/user_list_validation_%s.csv' % (week_ID, week_ID))

    #Format REG_DATE
    ulist['REG_DATE'] = ulist['REG_DATE'].apply(format_reg_date)

    #Segment the age
    ulist['0to30'] = ulist['AGE'].apply(age_function, age_low = 0, age_up = 30)
    ulist['30to50'] = ulist['AGE'].apply(age_function, age_low = 30, age_up = 50)
    ulist['50to100'] = ulist['AGE'].apply(age_function, age_low = 50, age_up = 100)

    list_age_bin = [col for col in ulist.columns.values if 'to' in col]
    ulist = ulist[['USER_ID_hash', 'PREF_NAME', 'SEX_ID', 'REG_DATE'] + list_age_bin]

    ulist =  ulist.T.to_dict().values()
    vec   = DictVectorizer(sparse = True)
    ulist = vec.fit_transform(ulist)
    #ulist is in csr format, make sure the type is int
    ulist = sps.csr_matrix(ulist, dtype = np.int32)

    #Save the matrix. They are already in csr format
    spi.mmwrite('../Data/Validation/%s/user_feat_mtrx_%s.mtx' % (week_ID, week_ID) , ulist)

def build_item_feature_matrix(week_ID):
    ''' Build item feature matrix

    arg : week_ID (str) validation week
    '''

    print 'Creating item_feature matrix for LightFM'

    def binarize_function(val, val_low =0, val_up = 100):
        '''Function to binarize a given column in slices
        '''
        if val_low<= val < val_up :
            return 1
        else :
            return 0

    #Utility to convert a date to the day of the week
    #(indexed by i in [0,1,..6])
    def get_day_of_week(row):
        '''Convert to unix time. Neglect time of the day
        '''
        row = row.split(' ')
        row = row[0].split('-')
        y,m,d = int(row[0]), int(row[1]), int(row[2])
        return date(y,m,d).weekday()

    #Load coupon data
    cpltr = pd.read_csv('../Data/Validation/%s/coupon_list_train_validation_%s.csv' % (week_ID, week_ID))
    cplte = pd.read_csv('../Data/Validation/%s/coupon_list_test_validation_%s.csv' % (week_ID, week_ID))

    cplte['DISPFROM_day'] = cplte['DISPFROM'].apply(get_day_of_week)
    cpltr['DISPFROM_day'] = cpltr['DISPFROM'].apply(get_day_of_week)
    cplte['DISPEND_day'] = cplte['DISPEND'].apply(get_day_of_week)
    cpltr['DISPEND_day'] = cpltr['DISPEND'].apply(get_day_of_week)

    cpltr['PRICE_0to50'] = cpltr['PRICE_RATE'].apply(binarize_function, val_low = 0, val_up = 30)
    cpltr['PRICE_50to70'] = cpltr['PRICE_RATE'].apply(binarize_function, val_low = 50, val_up = 70)
    cpltr['PRICE_70to100'] = cpltr['PRICE_RATE'].apply(binarize_function, val_low = 70, val_up = 100)

    cplte['PRICE_0to50'] = cplte['PRICE_RATE'].apply(binarize_function, val_low = 0, val_up = 30)
    cplte['PRICE_50to70'] = cplte['PRICE_RATE'].apply(binarize_function, val_low = 50, val_up = 51)
    cplte['PRICE_70to100'] = cplte['PRICE_RATE'].apply(binarize_function, val_low = 51, val_up = 100)

    list_quant_name = [0, 20, 40, 60, 80, 100]
    quant_step = list_quant_name[1] - list_quant_name[0]

    list_prices = cpltr['CATALOG_PRICE'].values
    list_quant = [np.percentile(list_prices, quant) for quant in list_quant_name]

    for index, (quant_name, quant) in enumerate(zip(list_quant_name, list_quant)) :
        if index>0 :
            cpltr['CAT_%sto%s' % (quant_name-quant_step, quant_name)] = cpltr['CATALOG_PRICE'].apply(binarize_function, val_low = list_quant[index-1], val_up = quant)
            cplte['CAT_%sto%s' % (quant_name-quant_step, quant_name)] = cplte['CATALOG_PRICE'].apply(binarize_function, val_low = list_quant[index-1], val_up = quant)

    list_prices = cpltr['DISCOUNT_PRICE'].values
    list_quant = [np.percentile(list_prices, quant) for quant in list_quant_name]
    for index, (quant_name, quant) in enumerate(zip(list_quant_name, list_quant)) :
        if index>0 :
            cpltr['DIS_%sto%s' % (quant_name-quant_step, quant_name)] = cpltr['DISCOUNT_PRICE'].apply(binarize_function, val_low = list_quant[index-1], val_up = quant)
            cplte['DIS_%sto%s' % (quant_name-quant_step, quant_name)] = cplte['DISCOUNT_PRICE'].apply(binarize_function, val_low = list_quant[index-1], val_up = quant)

    list_col_bin = [col for col in cplte.columns.values if 'to' in col]

    # List of features
    list_feat = ['GENRE_NAME', 'large_area_name', 'small_area_name', 'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE',
        'USABLE_DATE_WED', 'USABLE_DATE_THU', 'USABLE_DATE_FRI',
        'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
        'USABLE_DATE_BEFORE_HOLIDAY'] + list_col_bin

    #NA imputation
    cplte = cplte.fillna(-1)
    cpltr = cpltr.fillna(-1)

    list_col_to_str = ['PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPFROM_day', 'DISPEND_day', 'DISPPERIOD', 'VALIDPERIOD']
    cpltr[list_col_to_str] = cpltr[list_col_to_str].astype(str)
    cplte[list_col_to_str] = cplte[list_col_to_str].astype(str)

    # Reduce dataset to features of interest
    cpltr = cpltr[list_feat]
    cplte = cplte[list_feat]

    list_us = [col for col in list_feat if 'USABLE' in col]
    for col in list_us :
        cpltr.loc[cpltr[col]>0, col] = 1
        cpltr.loc[cpltr[col]<0, col] = 0
        cplte.loc[cpltr[col]>0, col] = 1
        cplte.loc[cpltr[col]<0, col] = 0

    # Binarize categorical features
    cpltr =  cpltr.T.to_dict().values()
    vec   = DictVectorizer(sparse = True)
    cpltr = vec.fit_transform(cpltr)
    cplte = vec.transform(cplte.T.to_dict().values())

    cplte = sps.csr_matrix(cplte, dtype = np.int32)
    cpltr = sps.csr_matrix(cpltr, dtype = np.int32)

    #Save the matrix. They are already in csr format
    spi.mmwrite('../Data/Validation/%s/train_item_feat_mtrx_%s.mtx' % (week_ID, week_ID), cpltr)
    spi.mmwrite('../Data/Validation/%s/test_item_feat_mtrx_%s.mtx' % (week_ID, week_ID), cplte)


if __name__ == '__main__':

    for week_ID in ['week51', 'week52']:

        #Preprocessing for similarity based recommander system
        create_LabelEncoded_files(week_ID)
        create_LabelBinarized_files(week_ID)
        var_choice = '1'
        prepare_similarity_data(var_choice, week_ID)
        #Preprocessing for a hybrid matrix factorisation method
        build_biclass_user_item_mtrx(week_ID)
        build_user_feature_matrix(week_ID)
        build_item_feature_matrix(week_ID)