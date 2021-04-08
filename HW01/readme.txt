本作業共使用了5個.py檔來實現:

1.MNIST.py
先以gzip這個套件來解壓縮MNIST dataset的壓縮檔，
並讀取解壓縮的檔案，最後會回傳已經處理好的data。

2.function.py
此py檔最主要為實作一些函數的功能，
function shuffle:Shuffle input lists data.
function ini_parameter:Initialize weights and biases of a layer.
function relu: Implement ReLU activation function and derivative of ReLU.
function softmax:Compute the softmax of input vector.
function cross_entropy_loss:Implement cross entropy loss function.
function lr_scheduler:Decay the learning rate every 25 epoch.
function compute_accuracy:This function does a forward pass of x, then checks if the indices
        		  of the maximum value in the output equals the indices in the label
		          y. Then it sums over each prediction and calculates the accuracy.
function plot_learning_curve:Plot the accuracy curve and loss curve for every epoch.
function plot_with_labels:Visualize last layer feature.

3.model.py
先產生parameter的矩陣並初始化，
功能有forward pass, backward propagation, update the parameters, save parameters and load parameters.

4.Training.py
分為三步驟:
1.Data preprocessing:
載入MNIST data，並將training data分成7:3的比例做為training data以及validation data，
在對label進行one-hot-encoding，最後Normalize image。

2.Training:
Hyper parameter:
Input_size = 28 * 28
learning rate=0.1
epoch = 100
Bactc size=64
a.learning rate初始值為0.1，而每訓練50個epoch會衰減。
b.每個epoch會先進行shuffle data，並以mini batch size方式餵入model進行訓練。
c.每個epoch訓練完後會對training data, validation data和testing data算出loss和accuracy，並且儲存。
d.每個epoch訓練完後會根據testing accuracy決定是否為最佳accuracy來儲存parameters。
e.每訓練50 epochs 會進行t-SNE來可視化validation data和testing data的結果。

3.訓練完成後會show出Loss curve和Accuracy curve。

5.test.py
功能為load最佳model，並對training data, validation data和testing data計算出loss和accuracy，
而計算出的loss和accuracy為最佳的，其中testing accuracy=98.03%。