.. index:: fix nve/spin
.. index:: fix nve/spin/kk

fix nve/spin command
====================

Accelerator Variants: *nve/spin/kk*

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID nve/spin keyword value ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* nve/spin = style name of this fix command
* zero or more keyword/value pairs may be appended
* keyword = *lattice_flag* or *spin_flag*

  .. parsed-literal::

       *lattice_flag* value = *yes* or *no*
         yes = integrate the positions and atomic velocities
         no = keep the lattice fixed
       *spin_flag* value = *yes* or *no*
         yes = integrate the spin vectors and spin velocities
         no = keep the spins fixed

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all nve/spin lattice_flag yes spin_flag yes
   fix 1 all nve/spin lattice_flag no

Description
"""""""""""

Perform constant NVE integration of Newtonian spin-lattice dynamics, in
which the spin vector :math:`\vec{S}_i = |\vec{S}_i| \, \vec{s}_i` of each
atom is a Cartesian coordinate with its own velocity
:math:`\vec{V}_{S,i}` and inertia :math:`I_i`, and its magnitude is not
constrained. Both sets of coordinates are advanced with velocity-Verlet:

.. math::

   \vec{V}_{S,i}(t + \Delta t/2) & = \vec{V}_{S,i}(t)
     + \frac{\Delta t}{2 I_i} \vec{F}_{S,i}(t) \\
   \vec{S}_i(t + \Delta t) & = \vec{S}_i(t)
     + \Delta t \, \vec{V}_{S,i}(t + \Delta t/2) \\
   \vec{V}_{S,i}(t + \Delta t) & = \vec{V}_{S,i}(t + \Delta t/2)
     + \frac{\Delta t}{2 I_i} \vec{F}_{S,i}(t + \Delta t)

and likewise for the positions and atomic velocities. After the drift the
spin vector is split back into the unit direction :math:`\vec{s}_i` and the
magnitude :math:`|\vec{S}_i|`, which atom style spin stores as *sp*. The
magnetic force :math:`\vec{F}_{S,i} = -\partial E / \partial \vec{S}_i`
is the Cartesian derivative of the energy, which the pair style must
declare that it computes.

The spin velocities and inertia are assigned with the *spin yes* and
*spmass* options of the *velocity create* command; atoms with zero
inertia keep their spins fixed. The *lattice_flag* and *spin_flag*
keywords select which of the two sets of coordinates is integrated.

The *nve/spin/kk* variant performs the same update on the device. In
double precision on CUDA devices its trajectories agree with the host
fix bit for bit, as long as the host compiler does not contract
multiply-adds into fused operations. It reports a non-finite spin
magnitude at the next thermodynamic output or at the end of the run
rather than in the step in which it occurs.

----------

.. include:: accel_styles.rst

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>`; the spin velocities and inertia are stored with the atoms.
None of the :doc:`fix_modify <fix_modify>` options are relevant to this
fix. No global or per-atom quantities are stored by this fix for access
by various :doc:`output commands <Howto_output>`. No parameter of this
fix can be used with the *start/stop* keywords of the :doc:`run <run>`
command. This fix is not invoked during :doc:`energy minimization
<minimize>`.

Restrictions
""""""""""""

This fix style can only be used if LAMMPS was built with the SPIN
package. See the :doc:`Build package <Build_package>` page for more
info.

It requires atom style spin, metal units, a single pair style (not a
hybrid style) that declares the Cartesian magnetic force, such as
dpa4spin/kk with *magnetic_force* set to *cartesian*, and spin
inertia assigned to at least one atom of the group when *spin_flag*
is *yes*. It cannot be combined with the fixes precession/spin or
setforce/spin, and it does not support
:doc:`run_style respa <run_style>`.

Related commands
""""""""""""""""

:doc:`atom_style spin <atom_style>`, :doc:`fix nve <fix_nve>`,
:doc:`velocity <velocity>`

Default
"""""""

The option defaults are lattice_flag = yes and spin_flag = yes.
