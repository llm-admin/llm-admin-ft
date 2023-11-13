# Program to display the Fibonacci sequence up to n-th term

nterms = 10
intN = 5
intK = 10
# first two terms
N, K = '5', '10'
count = 0

# check if the number of terms is valid
if nterms <= 0:
   print("Please enter a positive integer")
# if there is only one term, return n1
elif nterms == 1:
   print("Fibonacci sequence upto",nterms,":")
   print(n1)
# generate fibonacci sequence
else:
   print("Fibonacci sequence:")
   while count < nterms:
       #print(N)
       nth = N + K
       # update values
       N = K
       K = nth
       count += 1

print(K)
print(K[intN])
