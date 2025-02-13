#!/usr/bin/env python3

import tables as tb
f1 = tb.open_file('pytables.h5', 'w')
g1 = f1.create_group('/', 'g1')

a1 = f1.create_carray(g1, 'a1', tb.Int64Atom(), shape=(10000,))
