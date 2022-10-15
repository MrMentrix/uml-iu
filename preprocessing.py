import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Loading the data
df = pd.read_csv('data.csv')

"""This first section will be about re-naming feature names and turning the data into numerical data. Some features will be dropped."""

# renaming the columns for df, to make them easier to work with.
df.columns = [
    'self_employed',                # Are you self-employed?
    'employee_count',               # How many employees does your company or organization have?
    'tech_company',                 # Is your employer primarily a tech company/organization?
    'primary_tech_role',            # Is your primary role within your company related to tech/IT?
    'mh_benefits',                  # Does your employer provide mental health benefits as a part of healthcare coverage?
    'mh_care_options',              # Do you know the options for mental health care available under your employer-provided coverage?
    'mh_employer_discussion',       # Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?
    'mh_resources_provided',        # Does your employer offer resources to learn more about mental health concerns and options for seeking help?
    'mh_anonymity',                 # Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?
    'mh_leave',                     # If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:
    'mh_discuss_negative',          # Do you think that discussing a mental health disorder with your employer would have negative consequences?
    'ph_discuss_negative',          # Do you think that discussing a physical health issue with your employer would have negative consequences?
    'mh_discuss_coworker',          # Would you feel comfortable discussing a mental health disorder with your coworkers?
    'mh_discuss_supervisor',        # Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?
    'mh_ph_equal',                  # Do you feel that your employer takes mental health as seriously as physical health?
    'mh_neg_consq',                 # Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?
    'mh_coverage',                  # Do you have medical coverage (private insurance or state-provided) which includes treatment of mental health issues?
    'mh_resources',                 # Do you know local or online resources to seek help for a mental health disorder?
    'mh_diag_reveal_client',        # If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?
    'mh_reveal_neg_client',         # If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?
    'mh_diag_reveal_coworker',      # If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?
    'mh_reveal_neg_coworker',       # If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?
    'mh_productivity',              # Do you believe your productivity is ever affected by a mental health issue?
    'mh_time_affect',               # If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?
    'prev_employer',                # Do you have previous employers?
    'prev_mh_benefits',             # Have your previous employers provided mental health benefits?
    'prev_mh_care_options',         # Were you aware of the options for mental health care provided by your previous employers?
    'prev_employer_discussion',     # Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?
    'prev_mh_resources',            # Did your previous employers provide resources to learn more about mental health issues and how to seek help?
    'prev_mh_anonymity',            # Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?
    'prev_mh_discuss_negative',     # Do you think that discussing a mental health disorder with previous employers would have negative consequences?
    'prev_ph_discuss_negative',     # Do you think that discussing a physical health issue with previous employers would have negative consequences?
    'prev_mh_discuss_coworker',     # Would you have been willing to discuss a mental health issue with your previous co-workers?
    'prev_mh_discuss_supervisor',   # Would you have been willing to discuss a mental health issue with your direct supervisor(s)?
    'prev_mh_ph_equal',             # Did you feel that your previous employers took mental health as seriously as physical health?
    'prev_mh_neg_consq',            # Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?
    'ph_issue_interview',           # Would you be willing to bring up a physical health issue with a potential employer in an interview?
    'ph_interview_reason',          # Why or why not?
    'mh_issue_interview',           # Would you be willing to bring up a mental health issue with a potential employer in an interview?                
    'mh_interview_reason',          # Why or why not?.1
    'mh_hurts_career',              # Do you feel that being identified as a person with a mental health issue would hurt your career?
    'mh_coworkers_negatively',      # Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?
    'mh_share_family_friends',      # How willing would you be to share with friends and family that you have a mental illness?
    'mh_bad_response_workp',        # Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?
    'mh_observ_less_likely',        # Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?
    'family_mh_history',            # Do you have a family history of mental illness?
    'mh_disorder_past',             # Have you had a mental health disorder in the past?
    'mh_disorder_current',          # Do you currently have a mental health disorder?
    'condition_diagnose',           # If yes, what condition(s) have you been diagnosed with?
    'condition_belief',             # If maybe, what condition(s) do you believe you have?
    'diagnosed_professionally',     # Have you been diagnosed with a mental health condition by a medical professional?
    'professional_diagnose',        # If so, what condition(s) were you diagnosed with?
    'treatment_professional',       # Have you ever sought treatment for a mental health issue from a mental health professional?
    'treatment_interference',       # If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?
    'no_treatment_interference',    # If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?
    'age',                          # What is your age?
    'gender',                       # What is your gender?
    'country',                      # What country do you live in?
    'state',                        # What US state or territory do you live in?
    'work_country',                 # What country do you work in?
    'work_state',                   # What US state or territory do you work in?
    'work_position',                # Which of the following best describes your work position?
    'remote_work'                   # Do you work remotely?
]

binary = ["self_employed", "tech_company", "primary_tech_role", "mh_neg_consq", "mh_coverage", "prev_employer", "diagnosed_professionally", "treatment_professional"]
ordered_categorical = ["employee_count", "mh_leave", "mh_resources", "mh_time_affect", "prev_mh_resources", "mh_hurts_career", "mh_coworkers_negatively", "mh_share_family_friends", "mh_bad_response_workp", "treatment_interference", "no_treatment_interference", "remote_work", "prev_mh_anonymity"]
categorical = ["mh_benefits", "mh_care_options", "mh_employer_discussion", "mh_resources_provided", "mh_anonymity", "mh_discuss_negative", "ph_discuss_negative", "mh_discuss_coworker", "mh_discuss_supervisor", "mh_ph_equal", "mh_diag_reveal_client", "mh_reveal_neg_client", "mh_diag_reveal_coworker", "mh_reveal_neg_coworker", "mh_productivity", "prev_mh_benefits", "prev_mh_care_options", "prev_employer_discussion", "prev_mh_discuss_negative", "prev_ph_discuss_negative", "prev_mh_discuss_coworker", "prev_mh_discuss_supervisor", "prev_mh_ph_equal", "prev_mh_neg_consq", "ph_issue_interview", "mh_issue_interview", "mh_observ_less_likely", "family_mh_history", "mh_disorder_past", "mh_disorder_current", "country", "state", "work_country", "work_state"]
textual = ["ph_interview_reason", "mh_interview_reason", "condition_diagnose", "condition_belief", "professional_diagnose", "gender", "work_position"]
numerical = ["age"]

# Storing some information about the age so we can re-construct the age distribution later
# clearing some outliers (age < 20 or age > 80)
for i in range(len(df)):
    if df.loc[i, "age"] < 20 or df.loc[i, "age"] > 80:
        df.drop(i, inplace=True, axis=0)

age_range = df["age"].max() - df["age"].min() # saving this for later
age_min = df["age"].min()

# BINARY FEATURE

# dropping mh_coverage and primary_tech_role because of too many missing values
df.drop(columns=["mh_coverage", "primary_tech_role"], inplace=True)
binary.remove("mh_coverage")
binary.remove("primary_tech_role")

for feature in binary:    
    # if the feature is missing, remove the row, since I don't want to make assumptions about binary data
    df.dropna(subset=[feature], inplace=True, axis=0)

# re-indexing the dataframe
df.reset_index(drop=True, inplace=True)

# because of this, there is no data in mh_resources and mh_time_affect anymore, so I drop them
df.drop(columns=["mh_resources", "mh_time_affect"], inplace=True)
ordered_categorical.remove("mh_resources")
ordered_categorical.remove("mh_time_affect")


# turn all binary data into 0s and 1s
for feature in binary:
    df[feature] = df[feature].apply(lambda x: 1 if x == "Yes" or x == 1 else 0)

# CATEGORICAL FEATURES

# dropping location features because of many missing variables and because they don't seem to correlate with mental health
to_drop = ["country", "state", "work_country", "work_state"]
df.drop(columns=to_drop, inplace=True)
for feature in to_drop:
    categorical.remove(feature)

yes = ["Yes", "Yes, they all did", "Some did", "I was aware of some", "Yes, I was aware of all of them", "Some of them", "Yes, all of them", "Some of my previous employers", "Yes, at all of my previous employers"]
no = ["No", "No, none did", "No, I only became aware later", "None did", "None of them", "No, at none of my previous employers"]
unknown = ["Not eligible for coverage / N/A", "I don't know", "I am not sure", "Maybe", "N/A (not currently aware)"]

for feature in categorical:
    # if value in no, replace with 0, if value in yes, replace with 1, if value in unknown, replace with 0.5
    df[feature] = df[feature].apply(lambda x: 0 if x in no else (1 if x in yes else 0.5))

# ORDERED CATEGORICAL FEATURES

order = {
    "employee_count": {"1-5": 0, "6-25": 1, "26-100": 2, "100-500": 3, "500-1000": 4, "More than 1000": 5},
    "mh_leave": {"Very easy": 0, "Somewhat easy": 1, "Neither easy nor difficult": 2, "I don't know": 2, "Somewhat difficult": 3, "Very difficult": 4},
    "prev_mh_resources": {'None did': 0, 'Some did': 1, 'Yes, they all did': 2},
    "mh_hurts_career": {'No, it has not': 0, "No, I don't think it would": 1, 'Maybe': 2, 'Yes, I think it would': 3, 'Yes, it has': 4},
    "mh_coworkers_negatively": {'No, they do not': 0, "No, I don't think they would": 1, 'Maybe': 2, 'Yes, I think they would': 3, 'Yes, they do': 4},
    "mh_share_family_friends": {'Not open at all': 0, 'Somewhat not open': 1, 'Not applicable to me (I do not have a mental illness)': 2, 'Neutral': 2, 'Somewhat open': 3, 'Very open': 4},
    "mh_bad_response_workp": {'No': 0, 'Maybe/Not sure': 1, 'Yes, I experienced': 2, 'Yes, I observed': 2},
    "treatment_interference": {'Never': 0, 'Rarely': 1, 'Not applicable to me': 2, 'Sometimes': 3, 'Often': 4},
    "no_treatment_interference": {'Never': 0, 'Rarely': 1, 'Not applicable to me': 2, 'Sometimes': 3, 'Often': 4},
    "remote_work": {"Never": 0, "Sometimes": 1, "Always": 2},
    "prev_mh_anonymity": {"No": 0, "Sometimes": 1, "I don't know": 1, "Yes, always": 2}
}

for feature in ordered_categorical:
    # skip nan values
    df[feature] = df[feature].apply(lambda x: order[feature][x] if x in order[feature] else x)


# TEXTUAL FEATURES

# dropping all features besides gender and mental health diagnosis. Putting the gender into 4 categories: male/female/diverse/other
gender_dict = {
    "male": ["male", "m", "man", "cis male", "male (trans, ftm)", "fm", "male.", "male (cis)", "sex is male", "dude", "i'm a man why didn't you make this a drop down question. you should of asked sex? and i would of answered yes please. seriously how much text can this take?", "mail", "m|", "cisdude", "cis man"],
    "female": ["female", "I identify as female.", "female assigned at birth", "f", "woman", "cis female", "transitioned, m2f", "female or multi-gender femme", "female/woman", "cisgender female", "mtf", "fem", "female (props for making this a freeform field, though)", "cis-woman", "afab"],
    "divers": ["bigender", "non-binary", "genderfluid (born female)", "other/transfeminine", "androgynous", "male 9:1 female, roughly", "nb masculine", "genderqueer", "genderfluid", "enby", "malr", "genderqueer woman", "queer", "agender", "fluid", "male/genderqueer", "nonbinary", "genderflux demi-girl", "female-bodied; no feelings about gender", "transgender woman"],
    "other": ["other", "human", "none of your business", "unicorn"]
}
df["gender"] = df['gender'].apply(lambda x: 0 if str(x).strip().lower() in gender_dict["male"] else 1 if str(x).strip().lower() in gender_dict["female"] else 2 if str(x).strip().lower() in gender_dict["divers"] else 3)

# this way we keep the information about if a person believes they have a mental health condition
# set condition_belief to 1 if the value isn't nan
df["condition_belief"] = df["condition_belief"].apply(lambda x: 1 if not pd.isna(x) else 0)

textual.remove("condition_belief")
textual.remove("gender")

for feature in textual:
    df.drop(feature, axis=1, inplace=True)

# NUMERICAL FEATURES
for feature in numerical:
    # span all values in the range of 0 to 1
    df[feature] = df[feature].apply(lambda x: (x - df[feature].min()) / (df[feature].max() - df[feature].min()))

# IMPUTING MISSING VALUES

# impute missing values with the mode of the column
for feature in df.columns:
    if feature in binary:
        continue
    df[feature] = df[feature].apply(lambda x: df[feature].mode()[0] if pd.isnull(x) else x)

# imputing missing values in binary features with regression, based on all other features
for feature in binary:
    # create a copy of the dataframe without the feature we want to impute
    df_copy = df.drop(feature, axis=1)
    # create a new dataframe with only the feature we want to impute
    df_to_impute = df[[feature]]
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df_copy, df_to_impute, test_size=0.2, random_state=42)
    # train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # predict the missing values
    y_pred = model.predict(X_test)
    # replace the missing values with the predicted values
    df[feature] = df[feature].apply(lambda x: y_pred[0] if pd.isnull(x) else x)

print(df.shape)

# FEAUTRE CREATION

# I'll add some columns together, namely the ones about mental health diagnosis
# I'm basing this if the person currently has a disorder, believes they have one or has been diagnosed with one professionally
df['mh_issues'] = df.apply(lambda x: 1 if x["mh_disorder_current"] == 1 or x["condition_belief"] == 1 or x["diagnosed_professionally"] == 1 else 0, axis=1)

# 
df["age"] = df["age"].apply(lambda x: x * age_range + age_min)
df["gender"] = df["gender"].apply(lambda x: "male" if x == 0 else "female" if x == 1 else "divers" if x == 2 else "other")

# dropping the rows with "other" because there are only 6 of them and their answers to the gender-question haven't been very informative
df = df[df["gender"] != "other"]

# saving data to preprocessed.csv
df.to_csv("preprocessed.csv", index=False)