# rref
Reduced row echelon form solver for an augmented matrix (MATH2210: Applied Linear Algebra at UConn).

## Method
While looping through rows:
* Sort columns from least to most leading 0s
* Divide row by leading number to make it have a leading 1
* Subtract multiples of row from other rows to make column have all 0s except for leading 1

## Usage
Requirements:
* `numpy`: for matricies and operations
* `pytest` and `sympy`: for testing

Files:
* `rref.py`: contains `rref` method that calculated rref of augmented matrix
* `test_rref.py`: contains `pytest` tests for rref

## Testing
* Hand-calculated rrefs for multiple matricies
* Automated testing on `sympy` rrefs of random matricies
