<h3>When to use XGboosting</h3>
<ol>
    <li>When you have a large number of training Samples</li>
    <li>When you have a mixture of categorical and numeric features or just numeric features</li>
</ol>
<hr>
<h3>GridSearch Review</h3>
<ul>
    <li>Search exhaustively over a given set of hyperparameters,once per set of hyperparameters</li>
    <li>Number of models = number of distinct values perhyperparameter multiplied across each hyperparameter</li>
    <li>Pick final model hyperparameter values that give best cross-validated evaluation metric value</li>
</ul>
<hr>
<h3>RandomSearch Review</h3>
<ul>
    <li>Create a (possibly infinite) range of hyperparameter valuesper hyperparameter that you would like to search over</li>
    <li>Set the number of iterations you would like for the randomsearch to continue</li>
    <li>During each iteration, randomly draw a value in the range ofspecified values for each hyperparameter searched over andtrain/evaluate a model with those hyperparameters</li>
    <li>After you've reached the maximum number of iterations,select the hyperparameter configuration with the bestevaluated score</li>
</ul>