<h3>KMean Clustering</h3>
<hr>
<p>KMeand Clustering is one of the Centroid-based Clustering.K-means clustering is an unsupervised machine learning algorithm that groups unlabeled data into k number clusters, where k is a user-defined integer</p>

<span>How do we choose K value?</span>
<p>Most of the time, We visualize scatter plot. But in some cases, we use ELBOW METHOD to determine K value.</p>

<h5>Elbow Method</h5>
<li>
    <ul>make K values as array.</ul>
    <ul>In sklearn, you can get inerita by calling "model.inertia_". Store in array.</ul>
    <ul>plot kArray in X axis and inertiaArray in y axis</ul>
    <ul>the point where the decrease in inertia begins to slow is the best value for K</ul>
</li>