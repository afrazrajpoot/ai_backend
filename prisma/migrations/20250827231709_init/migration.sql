/*
  Warnings:

  - You are about to drop the column `department` on the `Employee` table. All the data in the column will be lost.
  - You are about to drop the column `email` on the `Employee` table. All the data in the column will be lost.
  - You are about to drop the column `manager` on the `Employee` table. All the data in the column will be lost.
  - You are about to drop the column `phone` on the `Employee` table. All the data in the column will be lost.
  - You are about to drop the column `position` on the `Employee` table. All the data in the column will be lost.
  - You are about to drop the column `salary` on the `Employee` table. All the data in the column will be lost.

*/
-- DropIndex
DROP INDEX "Employee_email_key";

-- AlterTable
ALTER TABLE "Employee" DROP COLUMN "department",
DROP COLUMN "email",
DROP COLUMN "manager",
DROP COLUMN "phone",
DROP COLUMN "position",
DROP COLUMN "salary";

-- AlterTable
ALTER TABLE "User" ADD COLUMN     "position" TEXT,
ALTER COLUMN "salary" SET DATA TYPE TEXT;
