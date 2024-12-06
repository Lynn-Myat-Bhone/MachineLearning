<p>We'll remove noise by smoothing with rolling windows.</p>
<ul>
    <li>First, take absolute value of each time point</li>
    <li>An then, Smooth by applying a rolling mean</li>
</ul>
<p>By calculating the envelope of each sound and smoothing it, you've eliminated much of the noise and have a cleaner signal to tell you what happening.</p>