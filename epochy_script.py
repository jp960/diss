def get_data():
    f = open("out_NYU_new.txt", "r")
    lines = f.readlines()
    all_epochs = list(str())
    sample_epochs = list()
    for line in lines:
        ind = lines.index(line)
        if line != "1!" and line != " [*] Reading checkpoint..." and line != " [*] Load SUCCESS" and line != "(1, 512, 256, 1)":
            bits = line.split(":")
            if bits[0] == "Epoch":
                all_epochs.append(bits[3].split(" ")[1])
            elif bits[0] == "[Sample] g_loss":
                sample_epochs.append(bits[1].split(" ")[1])
    i100 = [k * 100 - 1 for k in list(range(1, 101))]
    iepoch = [k for k in list(range(len(all_epochs)))]
    zipped_sample = list(zip(i100, sample_epochs))
    zipped_epoch = list(zip(iepoch, all_epochs))
    zipped_sample.sort(key=so)
    zipped_epoch.sort(key=so)
    f.close()
    return sample_epochs, all_epochs, zipped_sample, zipped_epoch


def so(val):
    return val[1]


if __name__ == '__main__':
    sample_epochs, all_epochs, zipped_sample, zipped_epoch = get_data()
    f = open("editted.txt", "w+")
    f.writelines("highest epoch: ")
    f.write(str(zipped_epoch[len(all_epochs)-1][1]) + "epoch: " + str(zipped_epoch[len(all_epochs)-1][0]))
    f.writelines("\n")
    f.writelines("lowest epoch: ")
    f.writelines(str(zipped_epoch[0][1]) + " epoch: " + str(zipped_epoch[0][0]))
    f.writelines("\n")
    f.writelines("highest sample: ")
    f.writelines(str(zipped_sample[len(sample_epochs)-1][1]) + " epoch: " + str(zipped_sample[len(sample_epochs)-1][0]))
    f.writelines("\n")
    f.writelines("lowest sample: ")
    f.writelines(str(zipped_sample[0][1]) + " epoch: " + str(zipped_sample[0][0]))
    f.close()
