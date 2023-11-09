
#HOME = /home/jakob

FC = gfortran

FCFLAGS = -O2 -g -fbounds-check
#FCFLAGS = -O2

PROGRAMS = test1_prog

all: $(PROGRAMS)

test1_prog: f90routines.o

test1_prog.o: f90routines.o

f90routines.o: 


%: %.o
	$(FC) $(FCFLAGS)  -o $@ $^

%.o: %.f90
	$(FC) $(FCFLAGS) -c $<

clean:
	rm -f *.o *.mod *.MOD

veryclean: clean
	rm -f *~ $(PROGRAMS)

