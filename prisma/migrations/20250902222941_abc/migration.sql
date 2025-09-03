/*
  Warnings:

  - Added the required column `risk_analysis` to the `IndividualEmployeeReport` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "IndividualEmployeeReport" ADD COLUMN     "risk_analysis" JSONB NOT NULL;

-- CreateTable
CREATE TABLE "Department" (
    "id" TEXT NOT NULL,
    "departments" JSONB NOT NULL,

    CONSTRAINT "Department_pkey" PRIMARY KEY ("id")
);
