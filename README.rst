Installation:
=============
.. code-block:: bash

   $ pip install -U git+https://github.com/tomerten/PyElegant.git

Known issues
============

Runing MPI from withing the singularity container for PElegant sometimes
gives errors like toe one below::

   Fatal error in PMPI_Init: Other MPI error, error stack:
   MPIR_Init_thread(565)..............: 
   MPID_Init(224).....................: channel initialization failed
   MPIDI_CH3_Init(105)................: 
   MPID_nem_init(324).................: 
   MPID_nem_tcp_init(178).............: 
   MPID_nem_tcp_get_business_card(425): 
   MPID_nem_tcp_init(384).............: gethostbyname failed, nbmachine01 (errno 1)

The way to solve this is to make sure that your terminal hostmname is the
same as the one in ``/etc/hosts/``::

   127.0.0.1    localhost
   127.0.0.1    terminal_host_name