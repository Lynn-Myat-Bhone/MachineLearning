<h3>Support Vector Classifiers </h3>
<p>SVC seem pretty cool becasue they can handle outliers and , becasue they allow missclassification ,
    they can handle overlapping classifications. But it cannot handle tons of overlapping data. That is where 
    " SUPPORT VECTOR MACHINE " come.
</p>
<hr>
<h3>Support Vector Machine </h3>
<span>SVM allows to use infinite-dimensional vector</span>
<p>IN SVM, it square the data and place in y-axis. 
    For example, data_point is in (0.5) in x_axis(1D). 
    SVM square the number 
    (0.5) squared  = 0.25
    then then data_point will be located in (0.5,0.25).
    after repeating for all data points.Now we can use SVC to classifications.
    if new data come in , square them  put it in x and y axis.
</p>
<span>The main ideas for Support Vector Machine are</span>
<ol>
    <li>start with relatively small dimensions</li>
    <li>move data into a higher dimensions (transform feature = square of (original feature))</li>
    <li>find a Support Vector Classifiers that seperates the higher dimensional data into two groups</li>

</ol>
<span>SVM use Kernel Math</span>