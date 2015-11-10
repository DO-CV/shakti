// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SHAKTI_PYTHON_NUMPY_HPP
#define DO_SHAKTI_PYTHON_NUMPY_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include <numpy/ndarrayobject.h>


inline
#if (PY_VERSION_HEX < 0x03000000)
void import_numpy_array()
#else
void * import_numpy_array()
#endif
{
  /* Initialise numpy API and use 2/3 compatible return */
  import_array();

#if (PY_VERSION_HEX >= 0x03000000)
  return NULL;
#endif
}


#endif /* DO_SHAKTI_PYTHON_NUMPY_HPP */
