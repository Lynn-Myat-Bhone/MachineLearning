<h3>Principal Component Analysis</h3>
<span>PCA is fundamental dimension reduction technique. It perfoms two steps</span>
<ul>
    <li>First step: decorrelation</li>
    <li>Second step: reduceds dimension by extraction</li>
</ul>

<h5>First step</h5>
<p>First you need check the correlation of features using "pearsonr()" method from "from scipy.stats import pearsonr"</p>
<span>It has values between -1 and 1 . -1 means total negative correlation. 0 mean no correlation. +1 mean total positive correlation</span>

<h5>Choosing a PCA Component</h5>
<ul>
    <li>Use the cumulative explained variance ratio to decide the number of components.</li>
    <li>Use a Fixed Number of Components(a fixed low-dimensional space for specific purposes: n_components=2 for 2D)</li>
    <li>Automatic Selection Using n_components='mle'(Maximum Likelihood Estimation)</li>
    <li>Scree Plot (Elbow Method)</li>

</ul>
