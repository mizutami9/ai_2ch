# 1(普通君).グーチョキパーを同じ確率で出すAI(33%)
# 2(ゼロヒャク君).絶対に勝つ『カチカク』と、絶対に負ける『マケカク』を同じ確率で出すAI(50%)
# 3(オールラウンダー君).カチカク、マケカク、グーチョキパーを同じ確率で出すAI(20%)
# 4(ドラちゃん).グーのみ
# 5(グチ君).グーチョキを同じ確率で出すAI(50%)
# 6(グパ君).グーパーを同じ確率で出すAI(50%)
# 7(パチ君).パーチョキを同じ確率で出すAI(50%)
# 8(コピー君).他のだれかの能力を一回戦ごとにコピーするAI

import random


# 0:グー　1：チョキ　2：パー　3：カチカク　4：マケカク
class person():
    def __init__(self, number):
        self.number = number

    def reach_out(self):
        if self.number != 7:
            mode = self.number
        else:
            mode = random.choice([0, 1, 2, 3, 4, 5, 6])

        if mode == 0:
            return self.out_0()
        elif mode == 1:
            return self.out_1()
        elif mode == 2:
            return self.out_2()
        elif mode == 3:
            return self.out_3()
        elif mode == 4:
            return self.out_4()
        elif mode == 5:
            return self.out_5()
        elif mode == 6:
            return self.out_6()

    def out_0(self):
        return random.choice([0, 1, 2])

    def out_1(self):
        return random.choice([3, 4])

    def out_2(self):
        return random.choice([0, 1, 2, 3, 4])

    def out_3(self):
        return random.choice([0])

    def out_4(self):
        return random.choice([0, 1])

    def out_5(self):
        return random.choice([0, 2])

    def out_6(self):
        return random.choice([1, 2])


# 0:あいこ　1:1の勝ち　2：2の勝ち
def result(te1, te2):
    if te1 == te2:
        return 0
    if te1 == 3 or te2 == 4:
        return 1
    if te1 == 4 or te2 == 3:
        return 2
    return (te2 - te1 + 3) % 3


def num2te(te):
    if te == 0:
        return "グー"
    elif te == 1:
        return "チョキ"
    elif te == 2:
        return "パー"
    elif te == 3:
        return "カチカク"
    else:
        return "マケカク"


def printer(te1, te2, res):
    if res == 0:
        reres = "あいこ"
    elif res == 1:
        reres = "左のかち"
    else:
        reres = "右のかち"

    print(num2te(te1) + " VS " + num2te(te2) + " " + reres)


def battle(ai_1, ai_2):
    res = 0
    while res == 0:
        te1 = ai_1.reach_out()
        te2 = ai_2.reach_out()
        res = result(te1, te2)
        printer(te1, te2, res)

    return res

def main():
    Alice = person(0)
    Bob = person(3)

    print(battle(Alice,Bob))

if __name__ == '__main__':
    main()