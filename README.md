---


---

<h1 id="global-wheat-head-detection-challenge">Global Wheat Head Detection Challenge</h1>
<h2 id="introduction">Introduction</h2>
<p>This project examines efficientDet’s performance in the use of global wheat head detection, which outperforms the state-of-art YOLO v3 and Faster RCNN. This function is believed to be useful when being used to estimate the density and size of wheat heads in different varieties and to assess health and maturity when making management decisions in their fields by farmers.</p>
<h2 id="eda">EDA</h2>
<p>The data set of this project is provided by multiple organizations. Thus there are variances within the images from brightness to style. This can be seen in the following image.<br>
<img src="https://github.com/MemphisMeng/Global-Wheat-Detection/blob/master/images/__results___46_1.png" alt="enter image description here"><br>
This should be noticed because I am going to apply ensemble learning so that I would also split the data.</p>
<h2 id="preprocessing">Preprocessing</h2>
<p>There are only approximately 1000 photos provided, which may not be sufficient to train a model. Here I used library Albumentations for the augmentation, which adds several effects to an image on the fly so that we can access to a large volume of dataset without collecting more actual data. The following image illustrates some effects that I am going to use:<br>
<img src="https://github.com/MemphisMeng/Global-Wheat-Detection/blob/master/images/__results___59_0.png" alt="enter image description here"></p>
<p>Here is how it looks when I combined all the selected images together:<br>
<img src="https://github.com/MemphisMeng/Global-Wheat-Detection/blob/master/images/__results___67_0.png" alt="enter image description here"></p>
<h2 id="data-split">Data Split</h2>
<p>As I mentioned before, there are photos collected by different organizations. To avoid biases generated by the data, the ideal splitting is to split the data without affecting its inside portions. Therefore I used Stratified K fold which remains the distribution of each organization in every single split.<br>
<img src="https://github.com/MemphisMeng/Global-Wheat-Detection/blob/master/images/__results___3_1.png" alt="enter image description here"><br>
As the figure shows, the overall disitrbution is almost the same as the one in one of the splits. Then what I did is implementing efficientDet in each split while flipping the images differently.</p>
<h2 id="ensemble-learning">Ensemble Learning</h2>
<p>After obtaing all the models trained from the previous steps, the final step is to ensemble them together and use the combined modelt to test on our test set. Our score is approximately 0.74 (top 10%).</p>
<h2 id="web-application">Web Application</h2>
<p>Our work has been deployed on cloud, feel free to have fun with it! (<a href="https://github.com/MemphisMeng/global-wheat-detection-web-app">https://github.com/MemphisMeng/global-wheat-detection-web-app</a>)<br>
Brief video:<br>
<img src="https://github.com/MemphisMeng/Global-Wheat-Detection/blob/master/video/explanation.gif" alt="enter image description here"></p>

