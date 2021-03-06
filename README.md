# DeepLearning

此為我曾做過的深度學習模型，分別為：

### 1. 手刻類神經網路：
建立若干層類神經網路，使用Back Propagation方法來優化各層參數。用以預測黑白手寫數字圖辨識。
### 2. 捲積神經網路(Convolution Neural Network)：
使用pytorch套件進行彩色圖像口罩分類辨識(有戴口罩、沒戴口罩、帶歪了的)，其中人臉抓取已用YOLO預先判定。
### 3. Recurrent Neural Network：
時序型資料的預測，使用若干國家若干時間的COVID-19確診人數資料進行訓練，基於過往確診人數預測未來確診人數是否會上升。該問題使用pytorch中LSTM架構解決。
### 4. Variational AutoEncoder(VAE)：
auto encoder的技巧，使用自身資料當作ground truth做類神經網路的訓練。期望可以將原始資料的特徵在中間層去萃取資訊(降維)，然後最大程度去還原原始資料。
### 5. GAN：
對抗式神經網路，做假人臉的生成(generator)以及人臉的判別(discriminator)
### 6. DQN：
強化式學習，以“FreewayDeterministic-v4”遊戲作為載體，訓練模型盡可能達到高分。
### 7. 報紙標題分類：
建立辭庫，將字詞轉換成dummy variable形式，使用LSTM架構再進行分類。

