from nlodatascience.generate_example_data import GenerateData
from nlodatascience.make_html import CreateHTML
import time
from nlodatascience import CreatingDataSet
from nlodatascience.ml_models import Models, EvaluateModels
from nlodatascience.logistic_regression_decomp import LogisticDecomposition
from sklearn.linear_model import LogisticRegression
from random import randint
import plotly.io as pio
pio.renderers.default = 'browser'

# Generate data
example_data = GenerateData(10000)
df_example = example_data.generate_dataset()

# Create Beslisboom
vars = ['leeftijd', 'geslacht', 'kanaal_instroom']
y = 'churn'  # de veriabelen waarvan de mean telkens wordt berekend
split_method = 'gini'  # de methode waarop gesplits kan worden (gini of mean)
min_records = 5000  # min N waarna nog een split gemaakt wordt
max_integer = 5  # maximaal aantal splits bij een integer variabelen
max_nr_splits = 2  # behalve voor categorische variabelen
min_split_values = 1000  # minimale N voor een split
nr_splits = {'leeftijd': 4}  # aantal splits per variabelen (kan overschreven worden door splits)
splits = {'leeftijd': [20, 25, 40, 60]}  # op welke waarde een split ,'besteding':[]
color_reverse = True  # Omkering kleuren. bij True, rood laag, blauw hoog
name_all = 'all players'
reorder = True  # False als de volgorde moet zijn zoals in vars, anders op meest impactvolle split op gini/mean
dir_file = '//'  # waar zoomable_multi_branch.html staat

create_html = CreateHTML(df_example, vars, y, split_method=split_method, min_records=min_records,
                         max_integer=max_integer, max_nr_splits=max_nr_splits, min_split_values=min_split_values,
                         nr_splits=nr_splits, splits=splits, color_reverse=color_reverse, name_all=name_all,
                         reorder=reorder, dir_file=dir_file)
start = time.process_time()
create_html.build_HTML('zoomable_multi_branch_test2.html', 'One Planet, Plant it',
                       explanation='Hier is mijn uitleg:</br> <span class="emoji">&#128514;</span>')
print(time.process_time() - start)

# MVL model
getting_data = CreatingDataSet(df_example, {})
subset_size = 10000
random_state = 12

X_vars = ['percentage_gelezen_mails', 'geslacht', 'leeftijd', 'maanden_lid', 'kanaal_instroom', 'actie_instroom',
          'contact_vorm']
y = 'tweede_merk_keuze_stl_spelers'

X_train, X_test, y_train, y_test = getting_data.get_train_test(y, X_vars, divided_by_max=False, scale_data=True,
                                                               add_random_int=False, add_random_cont=False, set_seed=2,
                                                               size=subset_size, test_size=0.25, random_state=12,
                                                               with_mean=True, with_std=True)
# Multinomial Logistic Regression Model
model = Models(X_train, y_train)
model_mvl = model.multivariate_logistic_regression()
evalueren_model = EvaluateModels(X_train, X_test, y_train, y_test)
evalueren_model.accuracy_models([model_mvl])
evalueren_model.visualize_probabilities_mvlogit(model_mvl)
evalueren_model.visualize_impact_variables(model_mvl, merken_x_as=False)

# Logistic Regression Model
subset_size = 5000
X_vars = ['percentage_gelezen_mails', 'geslacht', 'leeftijd', 'maanden_lid', 'kanaal_instroom', 'actie_instroom',
          'contact_vorm']
y = 'churn'


X_train, X_test, y_train, y_test = getting_data.get_train_test(y, X_vars, divided_by_max=False, scale_data=True,
                                                               add_random_int=False, add_random_cont=False, set_seed=2,
                                                               size=subset_size, test_size=0.25, random_state=12,
                                                               with_mean=True, with_std=True)


# Model maken en analyseren
model_churn = LogisticRegression().fit(X_train, y_train)
split_var = [randint(2018, 2022) for p in range(0, len(X_train))]
vars_to_show = []
model_decomp = LogisticDecomposition(model_churn)
bijdrage_tov_0_results = model_decomp.decomposition_logistic(X_train, split_var=split_var, plot=True, y=[],
                                                             X_vars_to_show=vars_to_show,
                                                             scaling_to_mean_odds_model=True,
                                                             scaling_to_actual_odds=False)
