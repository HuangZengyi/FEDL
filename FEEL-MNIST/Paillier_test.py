from phe import paillier
import gmpy2 as gy

public_key, private_key = paillier.generate_paillier_keypair()


def test_mul(encrypt_x, encrypt_y, origin_x, origin_y):
    # [[x]]*[[y]] = [[x+y]]
    encrypt_res = paillier.EncryptedNumber(public_key, encrypt_x.ciphertext() * encrypt_y.ciphertext())
    res = origin_x + origin_y
    assert res == private_key.decrypt(encrypt_res)


def test_exponent(encrypt_x, x, r):
    # [[x]]^r = [[x*r]]
    encrypt_res = paillier.EncryptedNumber(public_key, encrypt_x.ciphertext() ** r)
    res = x * r
    assert res == private_key.decrypt(encrypt_res)


secret_number_list = [3, 4, 5]
encrypted_number_list = [public_key.encrypt(x) for x in secret_number_list]

test_mul(encrypted_number_list[0], encrypted_number_list[1], secret_number_list[0], secret_number_list[1])
test_exponent(encrypted_number_list[0], secret_number_list[0], 3)

encrypt_res = encrypted_number_list[0] + encrypted_number_list[1]
assert 7 == private_key.decrypt(encrypt_res)

encrypt_res = encrypted_number_list[0] + 4
assert 7 == private_key.decrypt(encrypt_res)
