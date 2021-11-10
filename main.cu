#include "vars.h"

double timetotal;

int sendcount;
int recvcount;

double *sendbuff;
double *recvbuff;
double *sendbuff_d;
double *recvbuff_d;

#ifdef DIRECT
double *partbuff;
double *partbuff_d;
#endif

int myrank;
int numrank;
int numthread;

int main(int argc, char** argv) {

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&numrank);

  timetotal = omp_get_wtime();

  #pragma omp parallel
  if(omp_get_thread_num() == 0)
    numthread = omp_get_num_threads();

  //READ INPUT PARAMETERS
  char *chartemp;
  chartemp = getenv("SENDCOUNT");
  sendcount = atoi(chartemp);

  //REPORT INPUT PARAMETERS
  if(myrank == 0){
    printf("SEND COUNT: %d (%e GB)\n",sendcount,sendcount*sizeof(double)/1.e9);
    printf("\n");
    printf("NUMBER OF PROCESSES: %d\n",numrank);
    printf("NUMBER OF THREADS: %d\n",numthread);
    printf("\n");
  }

  int recvcount = sendcount/numrank;
  if(myrank < sendcount%numrank)
    recvcount++;

  //ALLOCATE CPU BUFFERS 
  sendbuff = new double[sendcount];
  for(int n = 0; n < sendcount; n++)
    sendbuff[n] = 1;
#ifdef COLLECTIVE 
  recvbuff = new double[recvcount];
  int recvcounts[numrank];
  MPI_Allgather(&recvcount,1,MPI_INT,recvcounts,1,MPI_INT,MPI_COMM_WORLD);
  if(myrank == 0)
    for(int p = 0; p < numrank; p++)
      printf("GPU %d RECVCOUNT: %d (%e GB)\n",p,recvcounts[p],recvcounts[p]*sizeof(double)/1.e9);
#endif
#ifdef DIRECT
  partbuff = new double[recvcount*numrank];
  recvbuff = new double[recvcount];
  int recvcounts[numrank];
  for(int p = 0; p < numrank; p++)
    recvcounts[p] = recvcount;
  int sendcounts[numrank];
  MPI_Allgather(&recvcount,1,MPI_INT,sendcounts,1,MPI_INT,MPI_COMM_WORLD);
  int recvdispl[numrank+1];
  int senddispl[numrank+1];
  recvdispl[0] = 0;
  senddispl[0] = 0;
  for(int p = 0; p < numrank; p++){
    recvdispl[p+1] = recvdispl[p]+recvcounts[p];
    senddispl[p+1] = senddispl[p]+sendcounts[p];
  }
#endif

  //ALLOCATE GPU BUFFERS
  cudaMallocHost((void**)&sendbuff_d,sendcount*sizeof(double));
#ifdef COLLECTIVE
  cudaMallocHost((void**)&recvbuff_d,recvcount*sizeof(double));
#endif
#ifdef DIRECT
  cudaMallocHost((void**)&partbuff_d,recvcount*numrank*sizeof(double));
  cudaMallocHost((void**)&recvbuff_d,recvcount*sizeof(double));
#endif
  cudaMemcpy(sendbuff_d,sendbuff,sendcount*sizeof(double),cudaMemcpyHostToDevice);

  //SYNCHRONIZED REDUCE-SCATTER
  if(myrank == 0)printf("\n");
  MPI_Barrier(MPI_COMM_WORLD);
  double timeEffective = omp_get_wtime();
  //DEVICE-TO-HOST MEMCPY
  MPI_Barrier(MPI_COMM_WORLD);
  double timeDeviceToHost = omp_get_wtime();
  cudaMemcpy(sendbuff,sendbuff_d,sendcount*sizeof(double),cudaMemcpyDeviceToHost);
  MPI_Barrier(MPI_COMM_WORLD);
  timeDeviceToHost = omp_get_wtime()-timeDeviceToHost;

#ifdef COLLECTIVE
  if(myrank == 0)printf("MPI COLLECTIVE\n");
  MPI_Barrier(MPI_COMM_WORLD);
  double timeReduceScatter = omp_get_wtime();
  MPI_Reduce_scatter(sendbuff,recvbuff,recvcounts,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  timeReduceScatter = omp_get_wtime()-timeReduceScatter;
#endif

#ifdef DIRECT
  if(myrank == 0)printf("DIRECT COMMUNICATIONS\n");
  MPI_Barrier(MPI_COMM_WORLD);
  double timeAlltoall = omp_get_wtime();
  MPI_Alltoallv(sendbuff,sendcounts,senddispl,MPI_DOUBLE,partbuff,recvcounts,recvdispl,MPI_DOUBLE,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  timeAlltoall = omp_get_wtime()-timeAlltoall;
#endif

  //HOST-TO-DEVICE MEMCPY
  MPI_Barrier(MPI_COMM_WORLD);
  double timeHostToDevice = omp_get_wtime();
#ifdef COLLECTIVE
  cudaMemcpy(recvbuff_d,recvbuff,recvcount*sizeof(double),cudaMemcpyHostToDevice);
#endif
#ifdef DIRECT
  cudaMemcpy(partbuff_d,partbuff,recvcount*sizeof(double)*numrank,cudaMemcpyHostToDevice);
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  timeHostToDevice = omp_get_wtime()-timeHostToDevice;

#ifdef DIRECT
  //GPU REDUCTION KERNEL
  MPI_Barrier(MPI_COMM_WORLD);
  double timeKernel = omp_get_wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  timeKernel = omp_get_wtime()-timeKernel;
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  timeEffective = omp_get_wtime()-timeEffective;

  //REPORT TIME
  if(myrank == 0){
    printf("\n");
    printf("EFFECTIVE REDUCE-SCATTER TIME: %e\n",timeEffective);
    printf("\n");
    printf("SYNCHRONIZED TIMES\n");
    printf("DEVICE-TO-HOST TIME: %e (%f GB/s)\n",timeDeviceToHost,sizeof(double)*sendcount/1.e9/timeDeviceToHost*numrank);
#ifdef COLLECTIVE
    printf("HOST-TO-DEVICE TIME: %e (%f GB/s)\n",timeHostToDevice,sizeof(double)*recvcount/1.e9/timeHostToDevice*numrank);
    printf("\n");
    printf("REDUCE-SCATTER TIME: %e\n",timeReduceScatter);
#endif
#ifdef DIRECT
    printf("HOST-TO-DEVICE TIME: %e (%f GB/s)\n",timeHostToDevice,sizeof(double)*recvcount/1.e9/timeHostToDevice*numrank);
    printf("ALL-TO-ALL TIME: %e (%f GB/s)\n",timeAlltoall,sizeof(double)*sendcount/1.e9/timeAlltoall*numrank);
    printf("\n");
    printf("KERNEL TIME: %e (%f GFLOPS)\n",timeKernel,timeKernel);
#endif
  }

  //REPORT ERROR
  cudaMemcpy(recvbuff,recvbuff_d,recvcount*sizeof(double),cudaMemcpyDeviceToHost);
  for(int n = 0; n < recvcount; n++)
    if(recvbuff[n] != numrank){
      printf("ERROR!\n");
      break;
    }

  MPI_Barrier(MPI_COMM_WORLD);
  if(myrank == 0)printf("TOTAL TIME: %e\n",omp_get_wtime()-timetotal);

  MPI_Finalize();

  return 0;
}
