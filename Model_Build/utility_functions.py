import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def bspn_roc_auc(df, identity, y_true, y_score):
    """
    Calculates the BSPN AUC metric used by the competition's leaderboard:
    BPSN (Background Positive, Subgroup Negative) AUC:
        Here, we restrict the test set to the non-toxic examples that
        mention the identity and the toxic examples that do not.
        A low value in this metric means that the model confuses
        non-toxic examples that mention the identity with toxic
        examples that do not, likely meaning that the model predicts
        higher toxicity scores than it should for non-toxic examples
        mentioning the identity.
    """
    bspn_mask = ((df[identity] > 0) & (df['target'] < .5)) \
                | ((df[identity] == 0) & (df['target'] >= .5))
    return roc_auc_score(y_true[bspn_mask], y_score[bspn_mask])


def bnsp_roc_auc(df, identity, y_true, y_score):
    """
    Calculates the BNSP AUC metric used by the competition's leaderboard:
    BNSP (Background Negative, Subgroup Positive) AUC:
        Here, we restrict the test set to the toxic examples that
        mention the identity and the non-toxic examples that do not.
        A low value here means that the model confuses toxic examples
        that mention the identity with non-toxic examples that do not,
        likely meaning that the model predicts lower toxicity scores than
        it should for toxic examples mentioning the identity.
    """
    bnsp_mask = ((df[identity] == 0) & (df['target'] < .5)) \
                | ((df[identity] > 0) & (df['target'] >= .5))
    return roc_auc_score(y_true[bnsp_mask], y_score[bnsp_mask])


def bnsp_and_bspn_per_id(df, y_true, y_score):
    """
    Gets BSPN and BNSP scores by ID
    """

    # List of columns in training set that flag mentions of identities
    identities = [
        'asian',
        'atheist',
        'bisexual',
        'black',
        'buddhist',
        'christian',
        'female',
        'heterosexual',
        'hindu',
        'homosexual_gay_or_lesbian',
        'intellectual_or_learning_disability',
        'jewish',
        'latino',
        'male',
        'muslim',
        'other_disability',
        'other_gender',
        'other_race_or_ethnicity',
        'other_religion',
        'other_sexual_orientation',
        'physical_disability',
        'psychiatric_or_mental_illness',
        'transgender',
        'white'
    ]

    # Initialise a dataframe to hold scores
    scores = pd.DataFrame({'identity': identities,
                           'bnsp_score': np.zeros([len(identities), ]),
                           'bspn_score': np.zeros([len(identities), ])
                           })

    # Loop through each identity and score
    for index, row in scores.iterrows():
        scores.loc[index, 'bnsp_score'] = bnsp_roc_auc(df, row['identity'], y_true, y_score)
        scores.loc[index]['bspn_score'] = bspn_roc_auc(df, row['identity'], y_true, y_score)

    return scores