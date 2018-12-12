from MulLayer import MulLayer

class AddLayer(object):
    def forward(self, x,y):
        out = x+y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return (dx, dy)

if __name__ == '__main__':
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_sum_layer = AddLayer()
    mul_tax_layer = MulLayer()

    #forward
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    sum_price = add_sum_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(sum_price, tax)

    #backward
    dprice = 1
    dsum_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price,dorange_price = add_sum_layer.backward(dsum_price)
    dorange,dorange_sum = mul_orange_layer.backward(dorange_price)
    dapple, dapple_sum = mul_apple_layer.backward(dapple_price)

    print dprice
    print dsum_price, dtax
    print dapple_price, dorange_price
    print dorange, dorange_sum
    print dapple, dapple_sum
