/*
  Warnings:

  - The `ingoing` column on the `Department` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - The `outgoing` column on the `Department` table would be dropped and recreated. This will lead to data loss if there is data in the column.

*/
-- AlterTable
ALTER TABLE "Department" DROP COLUMN "ingoing",
ADD COLUMN     "ingoing" JSONB[],
DROP COLUMN "outgoing",
ADD COLUMN     "outgoing" JSONB[];
