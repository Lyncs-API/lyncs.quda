#pragma once

namespace lyncs_quda {
    /*
     *  @brief Continous to EvenOdd
     *  @param[out] Same size as input but with sites sorted first even and then odd
     *  @param[in] A field of size prod(shape)*outer*inner
     *  @param[ndims] Size of shape
     *  @param[shape] Lattice shape to swap
     *  @param[outer] Degrees of freedom stored outside the lattice volume
     *  @param[inner] Degrees of freedom stored per lattice site (including sizeof(type))
     *  @param[swap] Swaps even with odd in the output/input
     */
    void evenodd(void* out, void* in, int ndims, int* shape, int outer=0, int inner=1, bool swap=0) {
      
      size_t volume = 1;
      for (int i=0; i<ndims; i++) {
	volume *= shape[i];
      }
      
      for(size_t v=0; v<volume; v++) {
	
	size_t tmp = v;
	bool isodd = swap;
	for (int i=ndims-1; i>=0; i--) {
	  isodd ^= tmp % shape[i] % 2;
	  tmp /= shape[i];
	}
	
	for(int i=0; i<outer; i++) {
	  memcpy((char*) out+(i*volume+isodd*(volume/2+volume%2)+v/2)*inner, (char*) in+(i*volume+v)*inner, inner);
	}
      }
    }

    /*
     *  @brief EvenOdd to Continous
     *  @param[out] Same size as input but with sites sorted in a continous order
     *  @param[in] A field of size prod(shape)*outer*inner
     *  @param[ndims] Size of shape
     *  @param[shape] Lattice shape to swap
     *  @param[outer] Degrees of freedom stored outside the lattice volume
     *  @param[inner] Degrees of freedom stored per lattice site (including sizeof(type))
     *  @param[swap] Swaps even with odd in the input
     */
    void continous(void* out, void* in, int ndims, int* shape, int outer=0, int inner=1, bool swap=0) {
      
      size_t volume = 1;
      for (int i=0; i<ndims; i++) {
	volume *= shape[i];
      }
      
      for(size_t v=0; v<volume; v++) {
	
	size_t tmp = v;
	bool isodd = swap;
	for (int i=ndims-1; i>=0; i--) {
	  isodd ^= tmp % shape[i] % 2;
	  tmp /= shape[i];
	}
	
	for(int i=0; i<outer; i++) {
	  memcpy((char*) out+(i*volume+v)*inner, (char*) in+(i*volume+isodd*(volume/2+volume%2)+v/2)*inner, inner);
	}
      }
    }
}

