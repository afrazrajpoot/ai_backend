/*
  Warnings:

  - Added the required column `employeeEmail` to the `Notification` table without a default value. This is not possible if the table is not empty.
  - Added the required column `employeeName` to the `Notification` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "Notification" ADD COLUMN     "employeeEmail" TEXT NOT NULL,
ADD COLUMN     "employeeName" TEXT NOT NULL;
