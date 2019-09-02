import os
from random import shuffle


def sort_image(image_dir):
    # 返回图像的编号，第一个和最后三个
    images = get_file_list(image_dir)
    indexs = []

    for image in images:
        index = int(image[-7:-4])  # 取三位数字作为图片序号

        indexs.append(index)

    indexs.sort()

    index_list = [indexs[0], indexs[-3], indexs[-2], indexs[-1]]
    return index_list


def fill8bits(num):
    num = str(num)
    while len(num) < 8:
        num = '0' + num
    return num


def find_slash(string):
    slash_index = []
    for i in range(0, len(string)):
        if string[i] == '/':
            slash_index.append(i)
    return slash_index


# 输入：文件夹
# 输出：(list)文件夹下所有的文件路径，不包含空文件夹
def get_file_list(dir):

    File_list = []
    # dirs是当前home下所有的目录；files是当前home下所有文件列表
    for home, dirs, files in os.walk(dir):

        for filename in files:

            if filename[0] is not '.':
                File_list.append(os.path.join(home, filename))

    return File_list


# 输入：label文件路径
# 输出：(int)情感标签
def read_label(filename):

    with open(filename, 'r') as f:
        label = f.readline()
    # 数字出现在第4位
    return int(label[3])


# 输入：图片文件夹根目录；情感标签的label文件路径
# 输出：(list)相应的图片序列的第一张图片（中性图片），和后三张图片（表情图片）路径
def get_image_list(home, file):
    Ipath_list = []

    slash_index = find_slash(file)
    pidx = file[slash_index[2]+1:slash_index[3]]
    eidx = file[slash_index[3]+1:slash_index[4]]

    image_dir = os.path.join(home, pidx, eidx)

    for imageIndex in sort_image(image_dir):
        Iname = pidx+'_'+eidx+'_'+fill8bits(imageIndex)+'.png'
        Ipath = os.path.join(image_dir, Iname)

        Ipath_list.append(Ipath)

    return Ipath_list


# 输入：txt文件路径
# 输出：(txt)按行打乱的txt文件路径
def shuffle_txt(fname):
    with open(fname) as f:
        lines = f.readlines()
        shuffle(lines)

    os.remove(fname)

    with open(fname, 'a+') as f:
        for line in lines:
            f.write(line)
    return 0


# 输入：情感标签根路径；图像根路径；产生的txt文件名
# 输出：(txt)按行打乱的所有数据的路径和label
# 默认值：'./CK+/Emotion'; './CK+/cohn-kanade-images'; './DataList.txt'
def get_full_data_list(emotion_root='./CK+/Emotion',
                       image_root='./CK+/cohn-kanade-images',
                       data_list='./DataList.txt'):
    emotion_file_list = get_file_list(emotion_root)

    for emotion_file in emotion_file_list:

        label = read_label(emotion_file)

        Ipath_list = get_image_list(image_root, emotion_file)

        with open(data_list, 'a+') as g:
            for i in range(len(Ipath_list)):
                Ipath = Ipath_list[i]

                if i == 0:
                    g.write(Ipath + '    ' + '0' + '\n')
                else:
                    g.write(Ipath + '    ' + str(label) + '\n')

    shuffle_txt(data_list)


# 输入：总图片路径的txt
# 输出：(10txt)分成10份的txt
# 默认值：'./CK+/Emotion'; './CK+/cohn-kanade-images'; './DataList.txt'
def split_into_k_fold(filename, k):
    with open(filename, 'r') as f:
        lines = f.readlines()

    ls = []
    for i in range(k):
        ls.append([])

    for i in range(len(lines)):
        line = lines[i]
        ls[i % k].append(line)

    for i in range(k):
        with open(str(i)+'.txt', 'a+') as f:
            lines2write = ls[i]
            for line2write in lines2write:
                f.write(line2write)


if __name__ == '__main__':
    split_into_k_fold('DataList.txt', 10)
