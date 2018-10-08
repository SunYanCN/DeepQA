def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, i in enumerate(input_list):

            if type(i) == list:
                input_list = i + input_list[index + 1:]
                break
            else:
                output_list.append(i)
                input_list.pop(index)
                break

    return output_list


def dereplication(input_list):
    return list(set(input_list))
