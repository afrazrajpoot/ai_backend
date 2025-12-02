/*
  Warnings:

  - You are about to drop the column `approvalStatus` on the `Company` table. All the data in the column will be lost.
  - You are about to drop the column `currentSeats` on the `Company` table. All the data in the column will be lost.
  - You are about to drop the column `maxSeats` on the `Company` table. All the data in the column will be lost.
  - You are about to drop the column `notes` on the `Company` table. All the data in the column will be lost.
  - You are about to drop the column `paymentStatus` on the `Company` table. All the data in the column will be lost.
  - You are about to drop the column `subscriptionEndDate` on the `Company` table. All the data in the column will be lost.
  - You are about to drop the column `subscriptionStartDate` on the `Company` table. All the data in the column will be lost.
  - You are about to drop the column `subscriptionTier` on the `Company` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "Company" DROP COLUMN "approvalStatus",
DROP COLUMN "currentSeats",
DROP COLUMN "maxSeats",
DROP COLUMN "notes",
DROP COLUMN "paymentStatus",
DROP COLUMN "subscriptionEndDate",
DROP COLUMN "subscriptionStartDate",
DROP COLUMN "subscriptionTier";

-- AlterTable
ALTER TABLE "User" ADD COLUMN     "amount" DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN     "quota" DOUBLE PRECISION DEFAULT 0.0;

-- DropEnum
DROP TYPE "CompanyApprovalStatus";

-- DropEnum
DROP TYPE "PaymentStatus";

-- DropEnum
DROP TYPE "SubscriptionTier";
