from sklearn.ensemble import RandomForestRegressor
from explainerdashboard import RegressionExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_fare, titanic_names

feature_descriptions = {
    "Sex": "Gender of passenger",
    "Gender": "Gender of passenger",
    "Deck": "The deck the passenger had their cabin on",
    "PassengerClass": "The class of the ticket: 1st, 2nd or 3rd class",
    "Fare": "The amount of money people paid", 
    "Embarked": "the port where the passenger boarded the Titanic. Either Southampton, Cherbourg or Queenstown",
    "Age": "Age of the passenger",
    "No_of_siblings_plus_spouses_on_board": "The sum of the number of siblings plus the number of spouses on board",
    "No_of_parents_plus_children_on_board" : "The sum of the number of parents plus the number of children on board",
}

X_train, y_train, X_test, y_test = titanic_fare()
train_names, test_names = titanic_names()
model = RandomForestRegressor().fit(X_train, y_train)
explainer = RegressionExplainer(model, X_test, y_test, 
       cats=['Deck', 'Embarked', 'Sex'],
       descriptions=feature_descriptions, # defaults to None
       idxs = test_names, # defaults to X.index
       index_name = "Passenger", # defaults to X.index.name
       target = "Fare", # defaults to y.name
       units = "$", # defaults to ""
                                )
db = ExplainerDashboard(explainer).run()
RegressionExplainer()