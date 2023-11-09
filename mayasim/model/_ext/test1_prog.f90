PROGRAM test

      USE f90routines

      IMPLICIT NONE

      INTEGER                           :: N, x_len, y_len, x, y, i, err
      REAL                              :: r
      REAL, DIMENSION(600, 600)         :: elevation, precipitation, water, flow
      INTEGER, DIMENSION(2,9)           :: neighbours
      REAL, PARAMETER :: PI = 3.1415927
      INTEGER, DIMENSION(2,282677)      :: land

      CALL RANDOM_NUMBER(precipitation)
      i=0
      x_len = 600
      y_len = 600
      DO x=1,x_len
        DO y=1,y_len
          elevation(x,y) = x+SIN(x/100*2*PI)
          r = (x - x_len/2)**2 + (y - y_len/2)**2
          IF(r<300**2) THEN
            i = i+1
            land(1,i) = x
            land(2,i) = y
          ENDIF
        END DO
      END DO
      DO i=1,100
        CALL f90waterflow(land,elevation,precipitation,neighbours,x_len,y_len,i,err,flow,water)
      END DO

END PROGRAM test
