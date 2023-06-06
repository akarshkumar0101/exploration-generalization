

with open('atari.txt') as f:
    lines57 = f.readlines()
lines57 = [line.strip() for line in lines57]
lines57 = [line.replace(' ', '') for line in lines57]
# print(lines57)

with open('atari_games.txt') as f:
    lines104 = f.readlines()
lines104 = [line.strip() for line in lines104]
lines104 = [line.lower() for line in lines104]
# print(lines104)


# print([line for line in lines57])
# print([line for line in lines57 if line not in lines104])
# print([line for line in lines104 if line not in lines57])
