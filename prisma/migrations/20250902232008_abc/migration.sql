/*
  Warnings:

  - Changed the type of `name` on the `Department` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.

*/
-- AlterTable
ALTER TABLE "Department" DROP COLUMN "name",
ADD COLUMN     "name" JSONB NOT NULL;
