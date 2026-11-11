.. index:: compute msd/spin
.. index:: compute msd/spin/mod

compute msd/spin command
========================

compute msd/spin/mod command
============================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID style keyword value ...

* ID, group-ID are documented in :doc:`compute <compute>` command
* style = *msd/spin* or *msd/spin/mod*
* zero or more keyword/value pairs may be appended
* keyword = *average*

  .. parsed-literal::

       *average* value = *yes* or *no*

Examples
""""""""

.. code-block:: LAMMPS

   compute 1 all msd/spin
   compute 2 all msd/spin/mod average yes

Description
"""""""""""

.. versionadded:: TBD

Define a computation that calculates the mean-squared displacement of
the spins of the group of atoms, the spin analogue of :doc:`compute msd
<compute_msd>`.  The spin of atom :math:`i` is the full vector
:math:`\vec{S}_i = |\vec{S}_i|\,\hat{s}_i` of :doc:`atom_style spin
<atom_style>`, whose magnitude evolves under the :doc:`TSPIN integrators
<fix_nve_tspin>`.

Style *msd/spin* calculates a vector of four quantities.  The first three
elements are the squared :math:`S_x`, :math:`S_y`, and :math:`S_z`
displacements of the spin vectors, summed and averaged over the atoms in
the group.  The fourth element is the total squared displacement
:math:`|\vec{S}_i - \vec{S}_i^{\,0}|^2`, summed and averaged over the
atoms in the group.

Style *msd/spin/mod* calculates the mean-squared displacement of the spin
magnitudes, :math:`(|\vec{S}_i| - |\vec{S}_i^{\,0}|)^2` summed and
averaged over the atoms in the group.

The displacement of a spin is from its reference value, which is the
spin at the time the compute command was issued unless the *average*
keyword is set to *yes*.  With *average yes* the reference is the running
average of the spin over all previous calls to the compute, including
the current call, as for the *average* keyword of :doc:`compute msd
<compute_msd>`, with the same dependence on the number of calls.

.. note::

   If you want the quantities calculated by these computes to be
   continuous when running from a :doc:`restart file <read_restart>`,
   use the same compute ID as in the original run, so that the fix
   these computes create to store the reference spins is initialized
   from the restart file.

Output info
"""""""""""

Style *msd/spin* calculates a global vector of length 4, which can be
accessed by indices 1--4, and style *msd/spin/mod* calculates a global
scalar.  See the :doc:`Howto output <Howto_output>` doc page for an
overview of LAMMPS output options.

The values are "intensive" and in units of the spin magnitude squared,
:math:`\mu_B^2` with :doc:`units metal <units>`.

Restrictions
""""""""""""

These computes are part of the SPIN package.  They are only enabled if
LAMMPS was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.

These computes require :doc:`atom_style spin <atom_style>` and cannot be
used with a dynamic group.

Related commands
""""""""""""""""

:doc:`compute msd <compute_msd>`,
:doc:`compute temp/tspin <compute_temp_tspin>`,
:doc:`fix nve/tspin <fix_nve_tspin>`

Default
"""""""

The option default is average = no.
