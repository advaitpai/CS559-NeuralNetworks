# put your image generator here
output = decoder(torch.rand(9,4))
# plt.subplot(3,3,0)
for i in range(0,len(output)):
    with torch.no_grad():
        plt.subplot(3,3,i+1)
        plt.title("Image "+str(i+1))
        plt.imshow(output[i].cpu().squeeze().numpy(), cmap='gist_gray')
        plt.axis('off')
plt.show()

# put your clustering accuracy calculation heres

from sklearn.cluster import KMeans # For clustering 
from sklearn.metrics import accuracy_score

training_set = torch.utils.data.DataLoader(train_data,batch_size=48000) # Loading the entire dataset as the batch
for img_data,label_data in training_set:
    with torch.no_grad():
        img_data = encoder(img_data)
    img_train = img_data.numpy()
    img_label = label_data.numpy().reshape(48000,1)
    print("Input Shape:",img_train.shape,"Label Shape:",img_label.shape)

total_clusters = len(np.unique(img_label))
print("Unique Clusters (should be 10 since digits 0-9):",total_clusters)

model = KMeans(n_clusters = total_clusters, random_state=2702,n_init=10)
print("Fitting Model ...........")
model.fit_transform(img_train) 
print("KMeans Model has been fit.")

kmeans_label = model.labels_
acc_score = accuracy_score(kmeans_label,img_label)
print("Accuracy Score before beginning of Index Reassignment:", acc_score)

for i in tqdm(range(0,int(len(kmeans_label)/10))):
    if (kmeans_label[i] != img_label[i]): # Index Swapping if the label does not match
        temp_label = kmeans_label
        correct_label = img_label[i] # True Value
        incorrect_label = kmeans_label[i] # Predicted Value
        for i in range(0,len(temp_label)): # Swapping indexes
            if(temp_label[i] == incorrect_label):
                temp_label[i] = correct_label
            elif(temp_label[i] == correct_label):
                temp_label[i] = incorrect_label
        temp_acc_score = accuracy_score(temp_label,img_label)
        if (temp_acc_score > acc_score):
            kmeans_label = temp_label
            acc_score = temp_acc_score
            print("Swap Occured!")
            print("Accuracy Score temp:",temp_acc_score)
            print("Accuracy score after swap:",accuracy_score(kmeans_label,img_label))
        
print("Final Accuracy Score:",acc_score)