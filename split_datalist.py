import os

#split scale
#for example
#real-total-40 and fake-total-20+20
#select-train-30+10+10 and validation and test is 5+5+5


data_list = './data_list/Img_data_c23_path.txt'
train_list_path = './data_list/Img_list_c23_train_path.txt'
val_list_path = './data_list/Img_list_c23_val_path.txt'
test_list_path = './data_list/Img_list_c23_test_path.txt'

train_img = []
with open(data_list) as read_file:
    for line in read_file:
        if "train" in line:
            train_img.append(line.strip())

val_img = []
with open(data_list) as read_file:
    for line in read_file:
        if "validation" in line:
            val_img.append(line.strip())

test_img = []
with open(data_list) as read_file:
    for line in read_file:
        if "test" in line:
            test_img.append(line.strip())

# write train_list
if os.path.exists(train_list_path):
    os.remove(train_list_path)

for idx in range(len(train_img)):
    with open(train_list_path, 'a+') as f:
        f.writelines(train_img[idx])
        f.write('\n')


# write val_list
if os.path.exists(val_list_path):
    os.remove(val_list_path)

for idx in range(len(val_img)):
    with open(val_list_path, 'a+') as f:
        f.writelines(val_img[idx])
        f.write('\n')

# write test_list
if os.path.exists(test_list_path):
    os.remove(test_list_path)

for idx in range(len(test_img)):
    with open(test_list_path, 'a+') as f:
        f.writelines(test_img[idx])
        f.write('\n')